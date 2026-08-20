#pragma once

// 后端抽象接口：QwenModel 只调用这个接口，不关心的 dtype、量化、硬件。
// 每个后端（CPU/CUDA/Metal/...）实现自己的 IBackend，注册到工厂。

#include <cstdint>

namespace tinyqwen {

    // 量化类型（与文件头 dtype 对应）
    enum class QuantType {
        kF32 = 0,   // FP32 权重
        kF16 = 1,   // FP16 权重
        kI4 = 2,    // INT4 权重量化（非对称 uint4，per-group scale+zero）
        // 未来扩展：kI8, kGPTQ, kAWQ, ...
    };

    // 权重张量的抽象：QwenModel 不关心的 dtype，只传这个结构体给后端
    struct WeightTensor {
        const void* data;          // 原始数据（fp32/fp16/int4/...）
        QuantType quant_type;      // 量化类型
        int rows;                  // 输出维度
        int cols;                  // 输入维度
        int group_size;            // 量化 group size（INT4/INT8 时 > 0）
        // 量化参数（scale/zero）已经打包在 data 里，后端自己解析
    };

    // 后端接口：所有算子的抽象
    class IBackend {
    public:
        virtual ~IBackend() = default;

        // ==== matvec 系列 ====
        // y = W @ x，W 是 WeightTensor（可能是 fp32/fp16/int4/...）
        virtual void matvec(const WeightTensor& w, const float* x, float* y,
                           int out_dim, int in_dim) = 0;

        // 成对 matvec：y1 = W1 @ x, y2 = W2 @ x（同一个 x）
        virtual void matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                                 const float* x, float* y1, float* y2,
                                 int out_dim, int in_dim) = 0;

        // QKV 三路融合：yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        virtual void matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                                const WeightTensor& wv, const float* x,
                                float* yq, float* yk, float* yv,
                                int q_dim, int kv_dim, int in_dim) = 0;

        // ==== GEMM 系列（prefill 用）====
        // y[M, N] = W[M, K] @ x[K, N]
        virtual void matmul(const WeightTensor& w, const float* x, float* y,
                           int M, int K, int N) = 0;

        // ==== 非 matvec 算子（都走 fp32）====
        virtual void rmsnorm(const float* x, const float* weight, float* y,
                            int n, float eps) = 0;

        virtual void rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta) = 0;

        virtual void partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                 int head_dim, int rotary_dim, int pos, float theta) = 0;

        virtual void attention_decode(const float* q, const float* k_cache,
                                     const float* v_cache, int seq_len, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     float scale, float* out) = 0;

        virtual void swiglu(float* gate, const float* up, int n) = 0;

        virtual int argmax(const float* logits, int n) = 0;

        virtual void top_k_logits(const float* logits, int n, int k,
                                 int* indices, float* values) = 0;

        // ==== GDN 专用算子（Qwen3.5）====
        virtual void causal_conv1d_update(float* x, float* state, const float* weight,
                                         float* out, int dim, int kernel_size) = 0;

        virtual void l2norm_inplace(float* x, int n, float eps) = 0;

        virtual void gdn_step(float* S, const float* q, const float* k, const float* v,
                             float g, float beta, float* o, int qk_dim, int v_dim) = 0;

        virtual void rmsnorm_gated(float* x, const float* gate, const float* weight,
                                  float* y, int n, float eps) = 0;
    };

} // namespace tinyqwen
