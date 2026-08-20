#pragma once

// QwenModel：把权重、KV cache、workspace 组装成一个能跑的模型，
// 并提供 forward_token() 做"输入一个 token、输出下一个 token"。
//
// 数学定义对照 docs/qwen_forward.md；整体流程概念见 docs/infra_primer.md。

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
    // top-k 结果：分数最高的 k 个 token 的 id 和对应 logits。
    struct TopKResult {
        std::vector<int> indices;
        std::vector<float> values;
    };

    // Qwen-like decoder-only 模型，支持两种架构：
    //   - Qwen2.x：所有层同构（full attention + SwiGLU）；
    //   - Qwen3.5：混合架构，linear_attention（Gated DeltaNet）与
    //     full_attention 按 full_attention_interval 交替（典型 3:1）。
    // batch = 1，token-by-token，greedy，fp32 reference 路径。
    //
    // 刻意不做"图/graph"抽象：forward 就是一个写死的函数，op 顺序和
    // docs/qwen_forward.md 一一对应，这样跟 PyTorch 对不上时能肉眼定位是哪步错了。
    // 通用图框架灵活但难调试，对"只跑一个模型"的我们来说是过度设计。
    class QwenModel {
    public:
        // 工厂函数：校验所有需要的 tensor 都在、形状对，然后初始化 KV cache 和 buffer。
        // max_seq_len：运行时 KV 容量上限，必须 <= 文件头里的 max_seq_len。
        // backend：计算后端（CPU/CUDA/...），如果为空则默认创建 CPU 后端。
        static bool create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out,
                           std::unique_ptr<IBackend> backend = nullptr);

        // 在当前位置（== kv_cache().seq_len()）上前向一个 token。
        // 返回 greedy 的下一个 token id；可选同时填 top-k logits。
        int forward_token(int token_id, TopKResult *topk = nullptr, int topk_k = 5);

        // 批量 prefill：一次处理 n 个 prompt token，线性投影走 GEMM。
        // 返回最后一个 token 的 greedy 下一 token id。
        int forward_prefill(const int *token_ids, int n, TopKResult *topk = nullptr, int topk_k = 5);

        void reset(); // 清空 KV cache 和 token 计数（开始新的一段对话）
        void set_prompt_len(int n) { prompt_len_ = n; } // 让 profiler 知道哪些是 prefill
        void set_fuse_gate_up(bool v) { fuse_gate_up_ = v; }
        void set_fuse_qkv(bool v) { fuse_qkv_ = v; }

        const ModelConfig &config() const { return cfg_; }
        Dtype dtype() const { return dtype_; } // 权重精度（f32 / f16）
        KvCache &kv_cache() { return kv_; }
        // GDN 状态总字节数（qwen35；qwen2 恒为 0）。
        size_t gdn_state_bytes() const { return gdn_state_.memory_bytes(); }
        int token_count() const { return token_count_; }
        // 最近一次 forward_token 的 logits（vocab_size 个 float，对词表每个词一个分数）。
        const float *last_logits() const { return logits_.data(); }

    private:
        QwenModel() = default;

        // 某一层的权重指针集合。都是裸指针（只读、不拥有），但分两类：
        //   - 大矩阵（proj/FFN）：const void*，dtype 跟随模型文件（f32/f16），
        //     forward 按 dtype_ 走 matvec_f32 / matvec_f16；
        //   - 小向量（norm/bias）：恒为 fp32 指针——f32 模型直接指向文件内存；
        //     f16 模型在 create() 时转成 fp32 副本存进 owned_f32_（量级 KB，
        //     换来 rmsnorm/bias 热点路径完全不改）。
        struct LayerWeights {
            const float *input_ln = nullptr; // attention 前的 RMSNorm 权重
            const void *q_proj = nullptr; // query 投影矩阵
            const void *k_proj = nullptr; // key 投影矩阵
            const void *v_proj = nullptr; // value 投影矩阵
            // Qwen2/2.5 的 attention 带 q/k/v bias（attention_bias=True）；o 无 bias。
            const float *q_bias = nullptr;
            const float *k_bias = nullptr;
            const float *v_bias = nullptr;
            const void *o_proj = nullptr; // 输出投影矩阵
            const float *post_ln = nullptr; // FFN 前的 RMSNorm 权重
            const void *gate = nullptr; // SwiGLU 的 gate 投影
            const void *up = nullptr; // SwiGLU 的 up 投影
            const void *down = nullptr; // SwiGLU 的 down 投影

            // --- v2 / Qwen3.5 full_attention 层专用 ---
            // q_proj 输出 2*q_dim（前半 query、后半输出门 gate，按 head 交错），
            // 在 create() 里绑成 [2*q_dim, hidden]。gate 用 sigmoid 乘到 attn 输出上。
            const float *q_norm = nullptr; // per-head RMSNorm（zero-centered，导出时已 +1）
            const float *k_norm = nullptr; // per-head RMSNorm（zero-centered，导出时已 +1）

            // --- v2 / Qwen3.5 linear_attention（Gated DeltaNet）层专用 ---
            const void *gdn_in_qkv = nullptr; // 混合 qkv 投影 [conv_dim, hidden]
            const void *gdn_in_z = nullptr; // 门控 z 投影 [value_dim, hidden]
            const void *gdn_in_b = nullptr; // beta 投影 [num_v_heads, hidden]
            const void *gdn_in_a = nullptr; // a（dt）投影 [num_v_heads, hidden]
            const void *gdn_out_proj = nullptr; // 输出投影 [hidden, value_dim]
            const float *gdn_conv_w = nullptr; // causal conv1d 权重 [conv_dim, kernel]
            const float *gdn_a_log = nullptr; // [num_v_heads]
            const float *gdn_dt_bias = nullptr; // [num_v_heads]
            const float *gdn_norm = nullptr; // GDN 输出门控 RMSNorm 权重 [v_head_dim]
        };

        // 按名字取 tensor 并校验 ndim/shape/dtype（== 文件 dtype）；
        // 失败时填 *err 并返回 nullptr。
        const TensorView *require_view(const ModelFile &file, const std::string &name,
                                       const std::vector<uint64_t> &shape, std::string *err);

        // 小向量绑定为 fp32：f32 模型直接返回文件内指针；f16 模型转换进
        // owned_f32_ 后返回副本指针（见 LayerWeights 注释）。
        const float *bind_f32_vector(const TensorView *t);

        // matvec 分派薄封装：按 dtype_ 调 matvec_f32 / matvec_f16 通用入口。
        void mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const;
        void mm(const void *w, const float *x, float *y, int M, int K, int N) const;
        void mv_pair(const void *w1, const void *w2, const float *x, float *y1, float *y2,
                     int out_dim, int in_dim) const;
        void mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                    float *yq, float *yk, float *yv, int q_dim, int kv_dim, int in_dim) const;

        ModelConfig cfg_{};
        Profiler *profiler_ = nullptr;
        std::unique_ptr<IBackend> backend_; // 计算后端（CPU/CUDA/...）
        bool fuse_gate_up_ = true;
        bool fuse_qkv_ = true;
        int prompt_len_ = 0;
        int token_count_ = 0;
        int q_dim_ = 0; // query 总维度 = n_heads * head_dim
        int kv_dim_ = 0; // key/value 总维度 = n_kv_heads * head_dim
        int max_seq_len_ = 0;

        // --- Qwen3.5 混合架构的派生维度（kQwen2 时全为 0）---
        int gdn_qk_dim_ = 0; // GDN key 总维度 = linear_num_qk_heads * linear_qk_head_dim
        int gdn_value_dim_ = 0; // GDN value 总维度 = linear_num_v_heads * linear_v_head_dim
        int gdn_conv_dim_ = 0; // 混合 qkv 维度 = 2 * gdn_qk_dim_ + gdn_value_dim_
        int rotary_dim_ = 0; // partial RoPE 旋转的维度 = head_dim * partial_rotary_factor

        Dtype dtype_ = Dtype::kF32; // 权重 dtype（= 文件 header dtype），全模型单一
        int group_size_ = 0; // INT4 量化 group size（kI4 时 > 0，否则 0）
        bool lm_head_is_f32_ = false; // I4 tied 时 lm_head=embed（fp32），需 f32 matvec

        const void *embed_ = nullptr; // 词嵌入表 [vocab, hidden]（dtype 随模型）
        const float *final_norm_ = nullptr; // 最后的 RMSNorm（恒 fp32，见 bind_f32_vector）
        const void *lm_head_ = nullptr; // 输出投影到词表；tied 时 == embed_
        // f16 模型专用：norm/bias 小向量的 fp32 副本（create 时一次转换）。
        // 注意：外层 vector 扩容只会移动 vector 对象本身，其堆内存（data()
        // 指向处）不动，所以已返回的 float* 不会悬空。
        std::vector<std::vector<float>> owned_f32_;
        std::vector<LayerWeights> layers_;
        KvCache kv_; // qwen35：只为 full attention 层分配
        GdnState gdn_state_; // qwen35：linear attention 层的递归/conv 状态

        // workspace buffer：forward 时反复使用的临时空间，create() 时一次分配，
        // 避免每个 token 都 new/delete（那会很慢且产生碎片）。
        std::vector<float> hidden_, normed_, q_, k_, v_, attn_, o_, gate_, up_, ffn_, logits_;
        // Qwen3.5 专用 workspace：
        //   q_full_  q_proj 原始输出（query 与 gate 按 head 交错，2*q_dim 个）
        //   q_gate_  full attention 输出门（解交错后的 gate，q_dim 个）
        //   mixed_   GDN 混合 qkv 投影 + conv 输出（gdn_conv_dim 个）
        //   z_       GDN 门控向量（gdn_value_dim 个）
        //   b_/a_    GDN 的 beta/a 标量投影（linear_num_v_heads 个）
        //   gdn_out_ GDN 递归输出（gdn_value_dim 个）
        std::vector<float> q_full_, q_gate_, mixed_, z_, b_, a_, gdn_out_;
    };
} // namespace tinyqwen
