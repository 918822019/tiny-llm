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
#include "expert_store.h"
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
    // -------------------------------------------------------------------------
    // dequant_i4_to_f32: 按组把 [M,K] INT4 权重反量化为 fp32（批量 prefill 用）
    //
    // 磁盘布局（每行每组）：[scale_fp16(2B) | zero_fp16(2B) | packed(K/2 B)]
    // 反量化语义（与导出器 / HF 一致）：value = (uint4 - zero) * scale
    // 暴露为自由函数以便单元测试直接对照（见 test_matvec_i4.cpp）。
    // -------------------------------------------------------------------------
    void dequant_i4_to_f32(const uint8_t *w, float *dst, int M, int K, int group_size);

    // -------------------------------------------------------------------------
    // matmul_i4_batched: 融合 W4A8 批量 matmul（Qwen3.5 批量 prefill 用）
    //
    // Y[M,N] = dequant(W_i4[M,K]) @ X[K,N]，一次算完 N 个 token（token 主序）。
    // 数值方案与 decode 的 sdot4 matvec 完全一致（W4A8：权重 (q-8)、激活对称
    // int8、逐组 scale/zero 修正），因此批量与逐 token 逐位可比。
    // 仅 aarch64+dotprod 可用，否则返回 false（调用方回退 dequant+sgemm）。
    // xq / ax_scale / xqsum 为调用方提供的可复用 workspace。
    // -------------------------------------------------------------------------
    bool matmul_i4_batched(const void *w, const float *x, float *y, int M, int K, int N,
                           int group_size, std::vector<int8_t> &xq,
                           std::vector<float> &ax_scale, std::vector<int32_t> &xqsum);

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
                           std::unique_ptr<IBackend> backend = nullptr,
                           bool kv_fp16 = false,
                           ExpertStore *expert_store = nullptr);

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
        // forward_ppl: 批量 prefill + 全位置 lm_head + 交叉熵，计算困惑度
        //
        // 对 token_ids[0..n-1] 做 teacher-forcing：位置 i 的 logits 预测
        // token_ids[i+1]，累加 log_softmax（i=0..n-2，共 n-1 个计分位置）。
        // 复用批量 prefill 的层计算（GEMM），仅末段对每个位置做 norm+lm_head+CE。
        //
        // 返回平均负对数似然 NLL = -Σlog p / count；*out_count 输出计分位置数。
        // PPL = exp(NLL) 由调用方计算。调用前需 reset()。
        // ---------------------------------------------------------------------
        double forward_ppl(const int *token_ids, int n, long *out_count);

        // ---------------------------------------------------------------------
        // forward_prefill_qwen35_batch: Qwen3.5 批量 prefill（GEMM 路径）
        //
        // 权重每层只读一遍（i4 按组反量化到 fp32 / f16 转换 / f32 直用），
        // 线性投影走 BLAS GEMM 摊薄到全部 token；GDN 递归与因果 attention
        // 保留逐 token 顺序扫描。详见 qwen_forward_prefill_qwen35.cpp 头注。
        //
        // 返回值: 末位 prompt token 的 greedy 下一 token；-2 = 无法处理
        // （平台无 GEMM 后端 / dtype 不支持），调用方回退逐 token 路径。
        // ---------------------------------------------------------------------
        int forward_prefill_qwen35_batch(const int *token_ids, int n,
                                         TopKResult *topk = nullptr, int topk_k = 5);

        // ---------------------------------------------------------------------
        // forward_prefill_moe_batch: MoE 批量 prefill（按专家分组）
        //
        // 逐 token prefill 的浪费：同一专家被多个 token 重复选中，却每次都重新
        // 读盘、各算一次 matvec（实测 n=32 时 bytes_read 57.4 GB、hits=0）。
        // 本路径改为按专家分组：router 算完全部 N token → 按专家归组 → 每专家
        // 只加载一次 → 对其所有 token 做一次 batch GEMM → 按 topk 权重散回。
        //
        // 批量：qkv_proj / o_proj / router / 专家 FFN（走 matmul_gptq）。
        // 顺序：rope / kv_append / causal attention（token i 只看 0..i）。
        //
        // 返回值: 末位 prompt token 的 greedy 下一 token；-2 = 无法处理
        // （专家非 GPTQ / n<=0），调用方回退逐 token 路径。
        // ---------------------------------------------------------------------
        int forward_prefill_moe_batch(const int *token_ids, int n,
                                      int *topk = nullptr, int topk_k = 0);

        // 批量 prefill 的最小 token 数（低于此值走逐 token）。
        // 反量化全部权重到 fp32 是一笔固定开销（单线程，~0.7s @4B），
        // 需要足够多的 token 才能摊平。4B 实测 crossover ≈ 30 token：
        // L=8/16/24 时批量反而慢（0.5-0.9×），L=33 起开始赚（1.1×），
        // 越长越赚（L=61 ≈1.8×）。阈值依据见优化日志。
        static constexpr int kBatchPrefillMinQwen35 = 32;

        // 开关：--no-batch-prefill 关闭（A/B 对照与回退用）
        void set_batch_prefill(bool v) { batch_prefill_enabled_ = v; }

        // PPL 累加器（forward_prefill 在 ppl_mode_ 下写入）
        bool ppl_mode_ = false;
        double ppl_sum_logprob_ = 0.0;
        long ppl_count_ = 0;

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

        // MoE：选择专家权重访问路径。false（默认）= resident 指针（专家常驻
        // ModelFile 内存，是正确性锚点）；true = SSD 卸载（经 ExpertStore pread +
        // LRU 缓存按需加载）。两者须逐位一致。
        void set_moe_ssd(bool v) { moe_ssd_ = v; }
        bool moe_ssd() const { return moe_ssd_; }
        // ExpertStore 命中/miss 统计（SSD 模式归因用）
        ExpertStoreStats moe_stats() const {
            return expert_store_ ? expert_store_->stats() : ExpertStoreStats{};
        }

        // ---- 属性访问器 ----

        // 模型配置（层数、维度、头数等）
        const ModelConfig &config() const { return cfg_; }

        // 权重精度（f32 / f16 / i4）
        Dtype dtype() const { return dtype_; }

        // KV cache 引用（用于外部访问 K/V 数据）
        KvCache &kv_cache() { return kv_; }

        // GDN 状态引用（Qwen3.5 专用）。Metal prefill 引擎在 GPU 上算完 GDN 递归后，
        // 必须把 recurrent/conv 状态写回这里，否则后续 CPU decode 会从零状态开始
        // （实测：prefill 首 token 正确但 decode 立刻发散）。
        GdnState &gdn_state() { return gdn_state_; }

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
        //   - Qwen2.x 层: 使用 input_ln, q/k/v/o_proj + q/k/v_bias, post_ln, gate/up/down
        //   - Qwen3 稠密层: 同上但无 q/k/v_bias，额外使用 q_norm, k_norm
        //   - Qwen3.5 full attention 层: 额外使用 q_norm, k_norm
        //   - Qwen3.5 linear attention 层: 使用 gdn_* 系列权重
        // ---------------------------------------------------------------------
        // ---------------------------------------------------------------------
        // RotParams: 单个子层的 BiIP 旋转参数（旋转量化模型专用）
        //   sign  — ±1 符号向量 [in_f]（fp32）；nullptr = 该子层未旋转
        //   scale — scaleH 向量 [in_f]（fp32）；nullptr = 被中和/无 scaleH
        //   block_size — Hadamard 分块大小（由 in_f 推导）
        // 推理时对激活施加配对逆变换：x → blockHadamard((x/scale)⊙sign)
        // ---------------------------------------------------------------------
        struct RotParams {
            const float *sign = nullptr;
            const float *scale = nullptr;
            int block_size = 0;
        };

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
            // attention 投影的真实 dtype。真 checkpoint 的 attention 投影是
            // bf16/fp16（不是 GPTQ），而模型级 dtype_ 是 kGPTQ4。若用 mv()
            // （按模型级 dtype）会把 fp16 当 GPTQ 解析 → 垃圾 → 乱码（坑 #19 同类）。
            Dtype attn_dtype = Dtype::kF16;

            // ---- FFN 前的 RMSNorm 权重 ----
            const float *post_ln = nullptr;  // post-attention RMSNorm（恒 fp32）

            // ---- SwiGLU FFN 投影矩阵 ----
            const void *gate = nullptr;  // gate 投影 [intermediate, hidden]（dtype 随模型）
            const void *up = nullptr;    // up 投影 [intermediate, hidden]（dtype 随模型）
            const void *down = nullptr;  // down 投影 [hidden, intermediate]（dtype 随模型）

            // ---- MoE（kQwen35MoE）专用：路由门 + 共享专家 + 路由专家 ----
            // MoE 层替换 dense FFN：路由门选 top-k 专家 + 共享专家（常驻）+
            // 路由专家（resident 指针 or ExpertStore SSD 卸载）。
            const void *moe_router = nullptr;       // 路由门 [n_experts, hidden]（resident）
            const void *moe_shared_gate = nullptr;  // 共享专家 gate [shared_inter, hidden]
            const void *moe_shared_up = nullptr;    // 共享专家 up   [shared_inter, hidden]
            const void *moe_shared_down = nullptr;  // 共享专家 down [hidden, shared_inter]
            // 共享专家门控 [1, hidden]：HF 的输出是 sigmoid(gate·x) * shared(x)。
            // 与 moe_shared_gate（FFN 的 gate_proj）是**两个不同的张量**。
            const void *moe_shared_gate_w = nullptr;
            // 门控张量自己的 dtype。不能复用 shared_dtype —— 两者可以不同
            // （fake 模型里共享专家是 GPTQ 而门控是 f32），复用会把 f32 当
            // GPTQ 解析直接崩（SIGBUS）。
            Dtype shared_gate_dtype = Dtype::kF16;
            // 共享专家的真实 dtype。真 checkpoint 的共享专家是 bf16/fp16（不是
            // GPTQ），而模型级 dtype_ 是 kGPTQ4。若用 mv()/mv_pair()（按模型级
            // dtype）会把 fp16 当 GPTQ 解析 → 垃圾 → 乱码（坑 #19 同类）。
            Dtype shared_dtype = Dtype::kF16;
            // 路由专家三块权重（resident 模式用；SSD 模式 moe_experts 为空，走 ExpertStore）
            struct ExpertPtrs { const void *gate; const void *up; const void *down; };
            std::vector<ExpertPtrs> moe_experts;

            // ---- QK per-head RMSNorm（Qwen3 稠密 / Qwen3.5 full_attention）----
            // Qwen2.x 没有这两个权重（nullptr）。语义按架构不同：Qwen3.5 是
            // zero-centered RMSNorm，导出时已折 +1；Qwen3 稠密是标准 RMSNorm，不折。
            const float *q_norm = nullptr;  // [head_dim]（恒 fp32）
            const float *k_norm = nullptr;  // [head_dim]（恒 fp32）

            // ---- v2 / Qwen3.5 full_attention 层专用 ----
            // q_proj 输出 2*q_dim（前半 query、后半输出门 gate，按 head 交错），
            // 在 create() 里绑成 [2*q_dim, hidden]。gate 用 sigmoid 乘到 attn 输出上

            // ---- v2 / Qwen3.5 linear_attention（Gated DeltaNet）层专用 ----
            const void *gdn_in_qkv = nullptr;   // 混合 qkv 投影 [conv_dim, hidden]
            const void *gdn_in_z = nullptr;     // 门控 z 投影 [value_dim, hidden]
            const void *gdn_in_b = nullptr;     // beta 投影 [num_v_heads, hidden]
            const void *gdn_in_a = nullptr;     // a（dt）投影 [num_v_heads, hidden]
            const void *gdn_out_proj = nullptr; // 输出投影 [hidden, value_dim]
            // GDN 投影的真实 dtype。真 checkpoint 的 GDN 投影是 bf16/fp16（不是
            // GPTQ），而模型级 dtype_ 是 kGPTQ4。若用 mv()（按模型级 dtype）会把
            // fp16 数据当 GPTQ 解析 → 垃圾 → 输出乱码（AGENTS.md 坑 #19 同类）。
            Dtype gdn_dtype = Dtype::kF16;
            const float *gdn_conv_w = nullptr;  // causal conv1d 权重 [conv_dim, kernel]
            const float *gdn_a_log = nullptr;   // 每个 v head 的 a 对数 [num_v_heads]
            const float *gdn_dt_bias = nullptr; // 每个 v head 的 dt bias [num_v_heads]
            const float *gdn_norm = nullptr;    // GDN 输出门控 RMSNorm 权重 [v_head_dim]

            // ---- BiIP 旋转参数（旋转量化模型；未旋转模型全为 nullptr）----
            RotParams rot_q, rot_k, rot_v, rot_o;      // attention 四个投影
            RotParams rot_gate, rot_up, rot_down;      // FFN 三个投影
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
        // W 是 const void*（可能 f32/f16/i4/gptq4），按 dtype_ 分派
        void mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const;

        // 当前权重 dtype 对应的 group_size：kGPTQ4 用 gptq_group_size_，
        // 其余（i4）用 quant_group_size_（group_size_）。
        int cur_group_size() const {
            return (dtype_ == Dtype::kGPTQ4) ? gptq_group_size_ : group_size_;
        }

        // 按张量**自身** dtype 做 matvec。mv() 用的是模型级 dtype_，对 dtype 与
        // master 不一致的张量会读错：GPTQ MoE 模型里 router（mlp.gate.weight）
        // 与非 tied 的 lm_head 是 fp32，若按 master=kGPTQ4 解读会把 fp32 数据当
        // GPTQ 块解析，router logits 全错 → 选错专家 → 输出全错且不报错。
        void mv_typed(const void *w, const float *x, float *y, int out_dim, int in_dim,
                      Dtype d, int group_size) const;

        // 旋转感知 matvec: 若 rot 有效，先对 x 施加 BiIP 配对旋转（进 rot_buf_），
        // 再做 matvec；rot.sign==nullptr 时等价于普通 mv（非旋转模型零开销分支）。
        void mv_rot(const void *w, const float *x, float *y, int out_dim, int in_dim,
                    const RotParams &rot) const;

        // 旋转感知 GEMM：若 rot 有效，先把输入批量 X 的每一列(每个 token)做 BiIP
        // 配对旋转，再做 GEMM；rot.sign==nullptr 时等价普通 mm。
        void mm_rot(const void *w, const float *x, float *y, int M, int K, int N,
                    const RotParams &rot) const;

        // GEMM: Y[M,N] = W[M,K] @ X[K,N]
        void mm(const void *w, const float *x, float *y, int M, int K, int N) const;

        // 成对 matvec: y1 = W1 @ x, y2 = W2 @ x
        void mv_pair(const void *w1, const void *w2, const float *x, float *y1, float *y2,
                     int out_dim, int in_dim) const;

        // 成对 matvec，按张量自身 dtype 路由（同 mv_typed 的理由）
        void mv_pair_typed(const void *w1, const void *w2, const float *x, float *y1,
                           float *y2, int out_dim, int in_dim, Dtype d, int group_size) const;

        // QKV 三路融合: yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        void mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                    float *yq, float *yk, float *yv, int q_dim, int kv_dim, int in_dim) const;

        // attention 统一分发：按 KV cache 精度选 fp32 / fp16-KV 融合 attention。
        // layer 为 KV cache 紧凑下标（full attention 层），seq 为有效位置数。
        void attention_kv(const float *q, int layer, int seq, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float *out) const;

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
        bool lm_head_is_f32_ = false; // tied 时 lm_head=embed（fp32），需走 f32 matvec
        bool lm_head_is_f16_ = false; // tied 时 lm_head=embed（fp16），需走 f16 matvec
        bool lm_head_is_i4_ = false;  // tied 时 lm_head=embed（紧凑 INT4），需走 i4 matvec
        // embed 张量的真实 dtype（可与文件级 dtype_ 不同，如 vq2 文件里 embed 存 f16）。
        // embed 查表与 tied-lm_head 投影都按它路由，而不是按文件级 dtype_。
        Dtype embed_dtype_ = Dtype::kF32;

        // router 与非 tied lm_head 的真实 dtype。GPTQ MoE 模型里这两个是 fp32
        // 而 master dtype 是 kGPTQ4，matvec 必须按它们各自的 dtype 路由。
        Dtype moe_router_dtype_ = Dtype::kF32;
        Dtype lm_head_dtype_ = Dtype::kF32;

        // 旋转量化标志：模型权重在旋转空间量化，推理需对激活做 BiIP 配对旋转。
        // 由是否存在 *.rot_sign 张量决定（见 create）。
        bool rotated_ = false;
        // 旋转激活的暂存缓冲（mv_rot 用，大小 = 最大 in_dim）。mutable：const 方法内可写。
        mutable std::vector<float> rot_buf_;

        // ==================== 全局权重 ====================

        const void *embed_ = nullptr;       // 词嵌入表 [vocab, hidden]（dtype 随模型）
        // embed 卸载时 embed_ 为 nullptr，改用 file_offset 按需 pread 单行。
        // embed 是查表（每 token 只读 1 行 = hidden*4 = 8 KB），却占 1187 MB fp32
        // （本模型 resident 的 41%）—— 留盘换内存，性能影响可忽略。
        uint64_t embed_file_offset_ = 0;    // 0 = 未卸载（走 embed_ 指针）
        std::vector<float> embed_row_;      // pread 单行的落点缓冲
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

        // ==================== Qwen3.5 批量 prefill workspace ====================
        // 全部 token 主序 [dim, N]（第 c 个 token 的向量在 c*dim）。
        // 按需扩容（resize 保留 capacity），跨 prefill 复用。
        struct BatchPrefillBufs {
            std::vector<float> hid;      // 残差流 [hidden, N]
            std::vector<float> normed;   // norm 后 [hidden, N]
            std::vector<float> out;      // o_proj / ffn 输出 [hidden, N]（复用）
            std::vector<float> mixed;    // GDN 混合 qkv [conv_dim, N]
            std::vector<float> z;        // GDN 门控 z [value_dim, N]
            std::vector<float> b;        // GDN beta 投影 [n_v_heads, N]
            std::vector<float> a;        // GDN a 投影 [n_v_heads, N]
            std::vector<float> gdn_out;  // GDN 递归输出 [value_dim, N]
            std::vector<float> q_full;   // q+gate 交错 [2*q_dim, N]
            std::vector<float> k;        // key [kv_dim, N]
            std::vector<float> v;        // value [kv_dim, N]
            std::vector<float> attn;     // attention 输出 [q_dim, N]
            std::vector<float> gate;     // FFN gate [inter, N]
            std::vector<float> up;       // FFN up [inter, N]
            std::vector<float> deq;      // 权重反量化 fp32 scratch [max M*K]
            // ---- 融合 W4A8 批量 matmul 的激活量化 workspace ----
            std::vector<int8_t> xq;      // int8 激活 [K, N]（token 主序）
            std::vector<float> ax_scale; // 每 token 激活量化 scale [N]
            std::vector<int32_t> xqsum;  // 每 token×组 激活和 [N * gpr]
        };
        BatchPrefillBufs bp_;
        bool batch_prefill_enabled_ = true;  // --no-batch-prefill 可关闭

        // ==================== MoE（kQwen35MoE）状态 ====================
        // 路由专家 dtype（fake 模型 expert=GPTQ；future 可 f32）；专家 FFN 经
        // 专门路径调用对应 kernel（不走 mv 的 master dtype 路由）。
        Dtype expert_dtype_ = Dtype::kF32;
        int moe_inter_ = 0;          // 路由专家 FFN 中间层宽度
        int shared_inter_ = 0;       // 共享专家 FFN 中间层宽度
        int n_experts_ = 0;          // 路由专家数
        int experts_per_tok_ = 0;    // top-k
        int gptq_group_size_ = 0;    // 专家 GPTQ 分组（expert_dtype_==kGPTQ4 时）
        bool moe_ssd_ = false;       // SSD 卸载模式开关
        ExpertStore *expert_store_ = nullptr;  // SSD 模式取专家权重（非拥有）

        // MoE workspace
        std::vector<float> moe_gate_logits_;   // 路由门输出 [n_experts]
        std::vector<float> moe_expert_gate_;    // 单专家 gate 支路 [moe_inter]
        std::vector<float> moe_expert_up_;      // 单专家 up 支路 [moe_inter]
        std::vector<float> moe_expert_out_;     // 单专家 down 输出 [hidden]
        std::vector<float> moe_ffn_acc_;        // FFN 累加器 [hidden]
        std::vector<float> moe_shared_out_;     // 共享专家 down 输出 [hidden]
        std::vector<int> moe_topk_idx_;         // 选中专家下标 [k]
        std::vector<float> moe_topk_w_;          // 归一化路由权重 [k]
    };
} // namespace tinyqwen