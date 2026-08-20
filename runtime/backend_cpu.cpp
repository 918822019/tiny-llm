// CPU 后端实现：调用 kernels/dispatch.h 的通用入口

#include "backend_cpu.h"
#include "../kernels/dispatch.h"
#include "../kernels/gdn_ops.h"  // partial_rope_ref

namespace tinyqwen {

    void CPUBackend::matvec(const WeightTensor& w, const float* x, float* y,
                           int out_dim, int in_dim) {
        if (w.quant_type == QuantType::kI4) {
            matvec_i4(static_cast<const uint8_t*>(w.data), x, y, out_dim, in_dim, w.group_size);
        } else if (w.quant_type == QuantType::kF16) {
            matvec_f16(static_cast<const uint16_t*>(w.data), x, y, out_dim, in_dim);
        } else {
            matvec_f32(static_cast<const float*>(w.data), x, y, out_dim, in_dim);
        }
    }

    void CPUBackend::matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                                 const float* x, float* y1, float* y2,
                                 int out_dim, int in_dim) {
        if (w1.quant_type == QuantType::kI4) {
            matvec_pair_i4(static_cast<const uint8_t*>(w1.data),
                          static_cast<const uint8_t*>(w2.data),
                          x, y1, y2, out_dim, in_dim, w1.group_size);
        } else if (w1.quant_type == QuantType::kF16) {
            matvec_pair_f16(static_cast<const uint16_t*>(w1.data),
                           static_cast<const uint16_t*>(w2.data),
                           x, y1, y2, out_dim, in_dim);
        } else {
            matvec_pair_f32(static_cast<const float*>(w1.data),
                           static_cast<const float*>(w2.data),
                           x, y1, y2, out_dim, in_dim);
        }
    }

    void CPUBackend::matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                                const WeightTensor& wv, const float* x,
                                float* yq, float* yk, float* yv,
                                int q_dim, int kv_dim, int in_dim) {
        if (wq.quant_type == QuantType::kI4) {
            matvec_qkv_i4(static_cast<const uint8_t*>(wq.data),
                         static_cast<const uint8_t*>(wk.data),
                         static_cast<const uint8_t*>(wv.data),
                         x, yq, yk, yv, q_dim, kv_dim, in_dim, wq.group_size);
        } else if (wq.quant_type == QuantType::kF16) {
            matvec_qkv_f16(static_cast<const uint16_t*>(wq.data),
                          static_cast<const uint16_t*>(wk.data),
                          static_cast<const uint16_t*>(wv.data),
                          x, yq, yk, yv, q_dim, kv_dim, in_dim);
        } else {
            matvec_qkv_f32(static_cast<const float*>(wq.data),
                          static_cast<const float*>(wk.data),
                          static_cast<const float*>(wv.data),
                          x, yq, yk, yv, q_dim, kv_dim, in_dim);
        }
    }

    void CPUBackend::matmul(const WeightTensor& w, const float* x, float* y,
                           int M, int K, int N) {
        if (w.quant_type == QuantType::kI4) {
            matmul_i4(static_cast<const uint8_t*>(w.data), x, y, M, K, N, w.group_size);
        } else if (w.quant_type == QuantType::kF16) {
            // f16 matmul 暂未实现，回退到 N 次 matvec
            for (int c = 0; c < N; ++c) {
                matvec_f16(static_cast<const uint16_t*>(w.data),
                          x + static_cast<size_t>(c) * K,
                          y + static_cast<size_t>(c) * M, M, K);
            }
        } else {
            matmul_f32(static_cast<const float*>(w.data), x, y, M, K, N);
        }
    }

    void CPUBackend::rmsnorm(const float* x, const float* weight, float* y,
                            int n, float eps) {
        ::tinyqwen::rmsnorm(x, weight, y, n, eps);
    }

    void CPUBackend::rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta) {
        ::tinyqwen::rope(q, k, n_heads, n_kv_heads, head_dim, pos, theta);
    }

    void CPUBackend::partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                 int head_dim, int rotary_dim, int pos, float theta) {
        // partial_rope_ref 在 ref_ops.h 里
        partial_rope_ref(q, k, n_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
    }

    void CPUBackend::attention_decode(const float* q, const float* k_cache,
                                     const float* v_cache, int seq_len, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     float scale, float* out) {
        ::tinyqwen::attention_decode(q, k_cache, v_cache, seq_len, max_seq_len,
                                    n_heads, n_kv_heads, head_dim, scale, out);
    }

    void CPUBackend::swiglu(float* gate, const float* up, int n) {
        ::tinyqwen::swiglu(gate, up, n);
    }

    int CPUBackend::argmax(const float* logits, int n) {
        return ::tinyqwen::argmax(logits, n);
    }

    void CPUBackend::top_k_logits(const float* logits, int n, int k,
                                 int* indices, float* values) {
        // top_k_logits 在 qwen_model.cpp 的匿名 namespace 里实现
        // 后端不提供此接口（它不是 kernel，是模型逻辑）
        (void)logits; (void)n; (void)k; (void)indices; (void)values;
    }

    void CPUBackend::causal_conv1d_update(float* x, float* state, const float* weight,
                                         float* out, int dim, int kernel_size) {
        ::tinyqwen::causal_conv1d_update(x, state, weight, out, dim, kernel_size);
    }

    void CPUBackend::l2norm_inplace(float* x, int n, float eps) {
        ::tinyqwen::l2norm_inplace(x, n, eps);
    }

    void CPUBackend::gdn_step(float* S, const float* q, const float* k, const float* v,
                             float g, float beta, float* o, int qk_dim, int v_dim) {
        ::tinyqwen::gdn_step(S, q, k, v, g, beta, o, qk_dim, v_dim);
    }

    void CPUBackend::rmsnorm_gated(float* x, const float* gate, const float* weight,
                                  float* y, int n, float eps) {
        ::tinyqwen::rmsnorm_gated(x, gate, weight, y, n, eps);
    }

    std::unique_ptr<IBackend> create_cpu_backend() {
        return std::make_unique<CPUBackend>();
    }

} // namespace tinyqwen
