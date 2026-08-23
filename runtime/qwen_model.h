#pragma once

// ============================================================================
// 文件: qwen_model.h
// 作用: QwenModel —— 把权重、KV cache、workspace 组装成一个能跑的模型，
//       并提供 forward_token() 做"输入一个 token、输出下一个 token"
//
// 核心设计:
//   QwenModel 是一个 decoder-only transformer 模型的推理引擎，支持:
//     - Qwen2.x: 所有层同构（full attention + SwiGLU）
//     - Qwen3.5: 混合架构，linear_attention（Gated DeltaNet）与
//       full_attention 按 full_attention_interval 交替（典型 3:1）
//
// 推理模式:
//   - batch = 1（单条序列推理）
//   - token-by-token（每次处理一个 token）
//   - greedy decoding（取 argmax 作为下一 token）
//   - fp32 reference 路径（作为数值基准）
//
// 设计哲学:
//   刻意不做"图/graph"抽象: forward 就是一个写死的函数，op 顺序和
//   docs/qwen_forward.md 一一对应。这样跟 PyTorch 对不上时能肉眼定位是哪步错了。
//   通用图框架灵活但难调试，对"只跑一个模型"的我们是过度设计。
//
// 数学定义: docs/qwen_forward.md
// 前置阅读: docs/infra_primer.md
// ============================================================================

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "gdn_state.h"
#include "kv_cache.h"
#include "model_loader.h"
#include "profiler.h"

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // TopKResult: top-k 结果
    //
    // 存储 logits 中前 k 个最大值对应的 token id 和分数。
    // 用于调试和展示 top-k 候选 token 及其概率。
    // -------------------------------------------------------------------------
    struct TopKResult {
        std::vector<int> indices;     // top-k 的 token id 列表
        std::vector<float> values;    // 对应的 logits 值列表
    };

    // -------------------------------------------------------------------------
    // QwenModel: Qwen-like decoder-only 模型推理引擎
    //
    // 支持两种架构:
    //   - Qwen2.x: 所有层同构（full attention + SwiGLU）
    //   - Qwen3.5: 混合架构，linear_attention（Gated DeltaNet）与
    //     full_attention 按 full_attention_interval 交替（典型 3:1）
    //
    // 推理约束:
    //   - batch = 1（单条序列）
    //   - token-by-token（每次处理一个 token）
    //   - greedy decoding（贪婪解码）
    //   - fp32 reference 路径（作为数值基准）
    //
    // 工厂创建:
    //   使用静态工厂方法 create() 创建实例，它会:
    //     1. 校验所有需要的 tensor 都在、形状对
    //     2. 初始化 KV cache 和 workspace buffers
    //     3. 绑定权重指针到 LayerWeights 结构
    // -------------------------------------------------------------------------
    class QwenModel {
    public:
        // ---------------------------------------------------------------------
        // create: 工厂函数 —— 创建 QwenModel 实例
        //
        // 参数:
        //   file:        已加载的 ModelFile（所有权不转移，但必须比 QwenModel 活得久）
        //   max_seq_len: 运行时 KV 容量上限，必须 <= 文件头里的 max_seq_len
        //   profiler:    性能分析器引用（用于记录每步耗时）
        //   err:         输出参数，失败时写入错误原因
        //   out:         输出参数，成功时指向创建的 QwenModel 实例
        //   backend:     计算后端（CPU/CUDA/...），如果为空则默认创建 CPU 后端
        //
        // 返回值:
        //   成功返回 true，失败返回 false 并把原因写进 *err
        //
        // 校验项目:
        //   - 所有必需的 tensor 都存在且形状正确
        //   - max_seq_len 不超过文件头中的值
        //   - 模型架构类型与文件内容一致
        // ---------------------------------------------------------------------
        static bool create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out,
                           std::unique_ptr<IBackend> backend = nullptr);

        // ---------------------------------------------------------------------
        // forward_token: 在当前位置上前向一个 token（decode 模式）
        //
        // 参数:
        //   token_id:   当前输入 token 的 id
        //   topk:       可选输出参数，填入 top-k logits 结果
        //   topk_k:     top-k 的数量（默认 5）
        //   need_logits: 是否计算 final_norm + lm_head + argmax（默认 true）。
        //                prefill 非末位 token 的 logits 会被丢弃，传 false 跳过
        //                这段（4B 上约省 16% 单 token 开销）；状态（KV/GDN）
        //                照常更新。跳过时返回 -1。
        //                ⚠️ --verbose / --dump-logits 的逐位置对照路径必须保持
        //                true（由 main 的直接调用保证，不走 forward_prefill）。
        //
        // 返回值:
        //   greedy 的下一个 token id（即 argmax(logits)）；need_logits=false 时 -1
        //
        // 说明:
        //   当前位置 = kv_cache().seq_len()。
        //   处理流程:
        //     1. embed(token_id) -> hidden
        //     2. 逐层 forward（attention + FFN）
        //     3. final_norm -> lm_head -> logits（need_logits 时）
        //     4. argmax(logits) 返回下一个 token id（need_logits 时）
        // ---------------------------------------------------------------------
        int forward_token(int token_id, TopKResult *topk = nullptr, int topk_k = 5,
                          bool need_logits = true);

        // ---------------------------------------------------------------------
        // forward_prefill: 批量 prefill —— 一次处理 n 个 prompt token
        //
        // 参数:
        //   token_ids: prompt token id 数组 [n]
        //   n:         prompt token 数量
        //   topk:      可选输出参数，填入 top-k logits 结果
        //   topk_k:    top-k 的数量（默认 5）
        //
        // 返回值:
        //   最后一个 token 的 greedy 下一 token id
        //
        // 说明:
        //   与 forward_token 的主要区别:
        //     - 线性投影走 GEMM（matmul）而非逐 token matvec，效率更高
        //     - 一次处理多个 token 的 attention 和 FFN
        //     - prefill 结束后 KV cache 中已有 n 个 token 的 K/V
        // ---------------------------------------------------------------------
        int forward_prefill(const int *token_ids, int n, TopKResult *topk = nullptr, int topk_k = 5);

        // ---------------------------------------------------------------------
        // reset: 清空 KV cache 和 token 计数
        //
        // 说明:
        //   开始新的一段对话时调用。清空 KV cache 和 GDN 状态，
        //   以及 token 计数和 prompt 长度标记。
        // ---------------------------------------------------------------------
        void reset();

        // 让 profiler 知道哪些是 prefill token（用于性能分析）
        void set_prompt_len(int n) { prompt_len_ = n; }

        // 设置是否融合 gate 和 up 投影（默认 true，性能优化）
        void set_fuse_gate_up(bool v) { fuse_gate_up_ = v; }

        // 设置是否融合 QKV 投影（默认 true，性能优化）
        void set_fuse_qkv(bool v) { fuse_qkv_ = v; }

        // ---- 属性访问器 ----

        // 模型配置（层数、维度、头数等）
        const ModelConfig &config() const { return cfg_; }

        // 权重精度（f32 / f16 / i4）
        Dtype dtype() const { return dtype_; }

        // KV cache 引用（用于外部访问 K/V 数据）
        KvCache &kv_cache() { return kv_; }

        // GDN 状态总字节数（Qwen3.5 专用；Qwen2.x 恒为 0）
        size_t gdn_state_bytes() const { return gdn_state_.memory_bytes(); }

        // 已处理的 token 总数（prefill + decode）
        int token_count() const { return token_count_; }

        // 最近一次 forward_token 的 logits 向量（vocab_size 个 float）
        // 对词表每个词一个分数，用于分析概率分布
        const float *last_logits() const { return logits_.data(); }

    private:
        // 默认构造函数，通过 create() 工厂方法创建实例
        QwenModel() = default;

        // ---------------------------------------------------------------------
        // LayerWeights: 某一层的权重指针集合
        //
        // 所有权说明:
        //   所有指针都是裸指针（只读、不拥有），分两类:
        //     - 大矩阵（proj/FFN）: const void*，dtype 跟随模型文件（f32/f16/i4），
        //       forward 按 dtype_ 走 matvec_f32 / matvec_f16 / matvec_i4；
        //     - 小向量（norm/bias）: 恒为 fp32 指针。f32 模型直接指向文件内存；
        //       f16 模型在 create() 时转成 fp32 副本存进 owned_f32_（量级 KB，
        //       换来 rmsnorm/bias 热点路径完全不改）。
        //
        // 架构说明:
        //   - Qwen2.x 层: 使用 input_ln, q/k/v/o_proj, post_ln, gate/up/down
        //   - Qwen3.5 full attention 层: 额外使用 q_norm, k_norm
        //   - Qwen3.5 linear attention 层: 使用 gdn_* 系列权重
        // ---------------------------------------------------------------------
        struct LayerWeights {
            // ---- attention 前的 RMSNorm 权重 ----
            const float *input_ln = nullptr;

            // ---- attention 投影矩阵 ----
            const void *q_proj = nullptr;  // query 投影 [q_dim, hidden]（dtype 随模型）
            const void *k_proj = nullptr;  // key 投影 [kv_dim, hidden]（dtype 随模型）
            const void *v_proj = nullptr;  // value 投影 [kv_dim, hidden]（dtype 随模型）

            // ---- Qwen2/2.5 的 attention bias（attention_bias=True）----
            // q/k/v 有 bias，o 无 bias
            const float *q_bias = nullptr;  // query bias [q_dim]（恒 fp32）
            const float *k_bias = nullptr;  // key bias [kv_dim]（恒 fp32）
            const float *v_bias = nullptr;  // value bias [kv_dim]（恒 fp32）

            // ---- attention 输出投影 ----
            const void *o_proj = nullptr;  // 输出投影 [hidden, q_dim]（dtype 随模型）

            // ---- FFN 前的 RMSNorm 权重 ----
            const float *post_ln = nullptr;  // post-attention RMSNorm（恒 fp32）

            // ---- SwiGLU FFN 投影矩阵 ----
            const void *gate = nullptr;  // gate 投影 [intermediate, hidden]（dtype 随模型）
            const void *up = nullptr;    // up 投影 [intermediate, hidden]（dtype 随模型）
            const void *down = nullptr;  // down 投影 [hidden, intermediate]（dtype 随模型）

            // ---- v2 / Qwen3.5 full_attention 层专用 ----
            // q_proj 输出 2*q_dim（前半 query、后半输出门 gate，按 head 交错），
            // 在 create() 里绑成 [2*q_dim, hidden]。gate 用 sigmoid 乘到 attn 输出上
            const float *q_norm = nullptr;  // per-head RMSNorm（zero-centered，导出时已 +1）
            const float *k_norm = nullptr;  // per-head RMSNorm（zero-centered，导出时已 +1）

            // ---- v2 / Qwen3.5 linear_attention（Gated DeltaNet）层专用 ----
            const void *gdn_in_qkv = nullptr;   // 混合 qkv 投影 [conv_dim, hidden]
            const void *gdn_in_z = nullptr;     // 门控 z 投影 [value_dim, hidden]
            const void *gdn_in_b = nullptr;     // beta 投影 [num_v_heads, hidden]
            const void *gdn_in_a = nullptr;     // a（dt）投影 [num_v_heads, hidden]
            const void *gdn_out_proj = nullptr; // 输出投影 [hidden, value_dim]
            const float *gdn_conv_w = nullptr;  // causal conv1d 权重 [conv_dim, kernel]
            const float *gdn_a_log = nullptr;   // 每个 v head 的 a 对数 [num_v_heads]
            const float *gdn_dt_bias = nullptr; // 每个 v head 的 dt bias [num_v_heads]
            const float *gdn_norm = nullptr;    // GDN 输出门控 RMSNorm 权重 [v_head_dim]
        };

        // ---------------------------------------------------------------------
        // require_view: 按名字取 tensor 并校验
        //
        // 参数:
        //   file:  ModelFile 引用
        //   name:  tensor 名字
        //   shape: 期望的形状（ndim 和每维大小）
        //   err:   输出参数，失败时写入错误原因
        //
        // 返回值:
        //   指向 TensorView 的指针；失败时填 *err 并返回 nullptr
        //
        // 校验项目:
        //   - tensor 是否存在
        //   - ndim 是否匹配
        //   - 各维度大小是否匹配
        //   - dtype 是否匹配（== 文件 dtype）
        // ---------------------------------------------------------------------
        const TensorView *require_view(const ModelFile &file, const std::string &name,
                                       const std::vector<uint64_t> &shape, std::string *err);

        // ---------------------------------------------------------------------
        // bind_f32_vector: 将小向量绑定为 fp32 指针
        //
        // 参数:
        //   t: 指向 TensorView 的指针
        //
        // 返回值:
        //   fp32 格式的 float* 指针
        //
        // 说明:
        //   - f32 模型: 直接返回文件内指针（零拷贝）
        //   - f16 模型: 转换进 owned_f32_ 后返回副本指针
        //   这是为了保持 rmsnorm/bias 等热点路径始终走 fp32，不需要分支判断
        // ---------------------------------------------------------------------
        const float *bind_f32_vector(const TensorView *t);

        // ---------------------------------------------------------------------
        // matvec 分派薄封装: 按 dtype_ 调对应的 dispatch 通用入口
        //
        // 说明:
        //   QwenModel 调用这些函数，它们内部根据 dtype_ 选择:
        //     - kF32: matvec_f32 / matvec_pair_f32 / ...
        //     - kF16: matvec_f16 / matvec_pair_f16 / ...
        //     - kI4:  matvec_i4 / matvec_pair_i4 / ...
        //   这样 forward 代码不需要分散 dtype 分支，保持简洁。
        // ---------------------------------------------------------------------

        // 单路 matvec: y = W @ x
        // W 是 const void*（可能 f32/f16/i4），按 dtype_ 分派
        void mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const;

        // GEMM: Y[M,N] = W[M,K] @ X[K,N]
        void mm(const void *w, const float *x, float *y, int M, int K, int N) const;

        // 成对 matvec: y1 = W1 @ x, y2 = W2 @ x
        void mv_pair(const void *w1, const void *w2, const float *x, float *y1, float *y2,
                     int out_dim, int in_dim) const;

        // QKV 三路融合: yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        void mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                    float *yq, float *yk, float *yv, int q_dim, int kv_dim, int in_dim) const;

        // ==================== 模型配置和状态 ====================

        ModelConfig cfg_{};                          // 模型配置（层数、维度等）
        Profiler *profiler_ = nullptr;               // 性能分析器
        std::unique_ptr<IBackend> backend_;          // 计算后端（CPU/CUDA/...）
        bool fuse_gate_up_ = true;                   // 是否融合 gate+up 投影
        bool fuse_qkv_ = true;                       // 是否融合 QKV 投影
        int prompt_len_ = 0;                         // prompt 长度（用于 profiler 区分 prefill/decode）
        int token_count_ = 0;                        // 已处理的 token 总数
        int q_dim_ = 0;                              // query 总维度 = n_heads * head_dim
        int kv_dim_ = 0;                             // key/value 总维度 = n_kv_heads * head_dim
        int max_seq_len_ = 0;                        // 运行时 KV 容量上限

        // ---- Qwen3.5 混合架构的派生维度（kQwen2 时全为 0）----
        int gdn_qk_dim_ = 0;     // GDN key 总维度 = linear_num_qk_heads * linear_qk_head_dim
        int gdn_value_dim_ = 0;  // GDN value 总维度 = linear_num_v_heads * linear_v_head_dim
        int gdn_conv_dim_ = 0;   // 混合 qkv 维度 = 2 * gdn_qk_dim_ + gdn_value_dim_
        int rotary_dim_ = 0;     // partial RoPE 旋转的维度 = head_dim * partial_rotary_factor

        // ==================== 权重精度和量化参数 ====================

        Dtype dtype_ = Dtype::kF32;   // 权重 dtype（= 文件 header dtype），全模型统一
        int group_size_ = 0;          // INT4 量化 group size（kI4 时 > 0，否则 0）
        bool lm_head_is_f32_ = false; // I4 tied 时 lm_head=embed（fp32），需特殊处理

        // ==================== 全局权重 ====================

        const void *embed_ = nullptr;       // 词嵌入表 [vocab, hidden]（dtype 随模型）
        const float *final_norm_ = nullptr; // 最后的 RMSNorm（恒 fp32，见 bind_f32_vector）
        const void *lm_head_ = nullptr;     // 输出投影到词表；tied 时 == embed_

        // f16 模型专用: norm/bias 小向量的 fp32 副本（create 时一次转换）
        // 注意: 外层 vector 扩容只会移动 vector 对象本身，其堆内存（data() 指向处）
        // 不动，所以已返回的 float* 不会悬空
        std::vector<std::vector<float>> owned_f32_;

        // ==================== 每层权重 ====================
        std::vector<LayerWeights> layers_;

        // ==================== 运行时缓存 ====================

        // KV cache: qwen3.5 只为 full attention 层分配
        KvCache kv_;

        // GDN 状态: qwen3.5 的 linear attention 层递归/conv 状态
        GdnState gdn_state_;

        // ==================== workspace buffers ====================
        //
        // forward 时反复使用的临时空间。create() 时一次分配，避免每个 token
        // 都 new/delete（那会很慢且产生碎片）。

        // Qwen2.x 和 Qwen3.5 共用
        std::vector<float> hidden_;  // 当前 token 的隐藏向量 [hidden_size]
        std::vector<float> normed_;  // RMSNorm 后的向量 [hidden_size]
        std::vector<float> q_;       // query 向量 [q_dim]（Qwen3.5 full attn: 解交错后）
        std::vector<float> k_;       // key 向量 [kv_dim]
        std::vector<float> v_;       // value 向量 [kv_dim]
        std::vector<float> attn_;    // attention 输出 [q_dim]
        std::vector<float> o_;       // 输出投影后的向量 [hidden_size]
        std::vector<float> gate_;    // gate 投影结果 [intermediate_size]
        std::vector<float> up_;      // up 投影结果 [intermediate_size]
        std::vector<float> ffn_;     // FFN 输出 [hidden_size]
        std::vector<float> logits_;  // logits 向量 [vocab_size]

        // Qwen3.5 专用 workspace
        std::vector<float> q_full_;  // q_proj 原始输出（query 与 gate 按 head 交错，2*q_dim）
        std::vector<float> q_gate_;  // full attention 输出门（解交错后的 gate，q_dim）
        std::vector<float> mixed_;   // GDN 混合 qkv 投影 + conv 输出（gdn_conv_dim）
        std::vector<float> z_;       // GDN 门控向量（gdn_value_dim）
        std::vector<float> b_;       // GDN 的 beta 标量投影（linear_num_v_heads）
        std::vector<float> a_;       // GDN 的 a（dt）标量投影（linear_num_v_heads）
        std::vector<float> gdn_out_; // GDN 递归输出（gdn_value_dim）
    };
} // namespace tinyqwen