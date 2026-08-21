// ============================================================================
// backend_cpu.cpp — CPU 后端实现
// ============================================================================
// 本文件实现 CPUBackend 类，它是推理引擎的 CPU 计算后端，实现了 IBackend
// 接口。所有算子通过调用 kernels/dispatch.h 的通用入口，由 dispatch 机制
// 根据运行时注册的实现名称分发到具体的 kernel（如 ref、neon 变体）。
//
// CPU 后端是默认后端，也是"参考实现"——所有其他后端的正确性都需要与
// CPU 后端的输出进行数值比对。
// ============================================================================

#include "backend_cpu.h"
#include "../kernels/dispatch.h"  // 通用算子分发入口
#include "../kernels/gdn_ops.h"   // partial_rope_ref

namespace tinyqwen {

    // 矩阵-向量乘法（单输出）：根据权重 dtype 分发到 f32/f16/i4 实现
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

    // 矩阵-向量乘法（双输出融合）：同时计算 y1=W1*x 和 y2=W2*x，x 只加载一次
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

    // 矩阵-向量乘法（三输出 QKV 融合）：同时计算 Q/K/V 三个投影，x 只加载一次
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

    // 批量矩阵乘法（GEMM）：用于 prefill 阶段一次性处理多个 token 的线性投影
    void CPUBackend::matmul(const WeightTensor& w, const float* x, float* y,
                           int M, int K, int N) {
        if (w.quant_type == QuantType::kI4) {
            matmul_i4(static_cast<const uint8_t*>(w.data), x, y, M, K, N, w.group_size);
        } else if (w.quant_type == QuantType::kF16) {
            // f16 matmul 暂未实现，回退到 N 次 matvec（效率较低）
            for (int c = 0; c < N; ++c) {
                matvec_f16(static_cast<const uint16_t*>(w.data),
                          x + static_cast<size_t>(c) * K,
                          y + static_cast<size_t>(c) * M, M, K);
            }
        } else {
            matmul_f32(static_cast<const float*>(w.data), x, y, M, K, N);
        }
    }

    // RMS 归一化：y = x / sqrt(mean(x^2) + eps) * weight
    void CPUBackend::rmsnorm(const float* x, const float* weight, float* y,
                            int n, float eps) {
        ::tinyqwen::rmsnorm(x, weight, y, n, eps);
    }

    // 旋转位置编码（全头 RoPE）：对 Q/K 的每个头做旋转，注入位置信息
    void CPUBackend::rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta) {
        ::tinyqwen::rope(q, k, n_heads, n_kv_heads, head_dim, pos, theta);
    }

    // 部分旋转位置编码（Qwen3.5）：只旋转每头前 rotary_dim 维，其余不变
    void CPUBackend::partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                 int head_dim, int rotary_dim, int pos, float theta) {
        partial_rope_ref(q, k, n_heads, n_kv_heads, head_dim, rotary_dim, pos, theta);
    }

    // 单步 attention 解码：Q 对 KV cache 中所有历史位置做加权求和
    void CPUBackend::attention_decode(const float* q, const float* k_cache,
                                     const float* v_cache, int seq_len, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     float scale, float* out) {
        ::tinyqwen::attention_decode(q, k_cache, v_cache, seq_len, max_seq_len,
                                    n_heads, n_kv_heads, head_dim, scale, out);
    }

    // SwiGLU 激活：gate[i] = gate[i] * sigmoid(gate[i]) * up[i]，gate 就地复用
    void CPUBackend::swiglu(float* gate, const float* up, int n) {
        ::tinyqwen::swiglu(gate, up, n);
    }

    // 取最大值索引：用于 greedy 采样
    int CPUBackend::argmax(const float* logits, int n) {
        return ::tinyqwen::argmax(logits, n);
    }

    // top-k logits：CPU 后端不提供此接口（是模型逻辑，非 kernel）
    void CPUBackend::top_k_logits(const float* logits, int n, int k,
                                 int* indices, float* values) {
        (void)logits; (void)n; (void)k; (void)indices; (void)values;
    }

    // 因果一维卷积单步更新（GDN）：每个通道独立做 causal conv1d
    void CPUBackend::causal_conv1d_update(float* x, float* state, const float* weight,
                                         float* out, int dim, int kernel_size) {
        ::tinyqwen::causal_conv1d_update(x, state, weight, out, dim, kernel_size);
    }

    // 就地 L2 归一化（GDN）：x = x / sqrt(sum(x^2) + eps)
    void CPUBackend::l2norm_inplace(float* x, int n, float eps) {
        ::tinyqwen::l2norm_inplace(x, n, eps);
    }

    // GDN delta rule 递归步：S = g*S + outer(k,v); o = beta * S^T * q
    void CPUBackend::gdn_step(float* S, const float* q, const float* k, const float* v,
                             float g, float beta, float* o, int qk_dim, int v_dim) {
        ::tinyqwen::gdn_step(S, q, k, v, g, beta, o, qk_dim, v_dim);
    }

    // 门控 RMSNorm（GDN）：y = rmsnorm(x) * weight * silu(gate)
    void CPUBackend::rmsnorm_gated(float* x, const float* gate, const float* weight,
                                  float* y, int n, float eps) {
        ::tinyqwen::rmsnorm_gated(x, gate, weight, y, n, eps);
    }

    // CPU 后端工厂函数
    std::unique_ptr<IBackend> create_cpu_backend() {
        return std::make_unique<CPUBackend>();
    }

} // namespace tinyqwen