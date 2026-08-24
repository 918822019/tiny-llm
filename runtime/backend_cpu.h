#pragma once

// ============================================================================
// 文件: backend_cpu.h
// 作用: CPU 后端实现 —— 包装 kernels/dispatch.h 的通用入口，实现 IBackend 接口
//
// 设计说明:
//   CPUBackend 是 IBackend 的 CPU 实现。它不直接实现任何算子，而是将所有调用
//   转发到 kernels/dispatch.h 中定义的通用入口函数。这些通用入口函数再根据
//   运行时选择的实现（ref/neon_mt 等）分发到具体的 kernel。
//
// 这是"保留 base + 可插拔优化"架构的关键一环:
//   QwenModel -> IBackend -> CPUBackend -> dispatch 通用入口 -> 具体 kernel
//
// 与其他后端的关系:
//   - CPUBackend: 通过 dispatch 层分发，支持 ref/NEON/多线程等变体
//   - CUDABackend: 每个算子单独调用 CUDA kernel（独立实现）
//   - 其他后端（Metal/OpenCL）: 类似模式，继承 IBackend 实现
// ============================================================================

#include "backend.h"
#include <memory>

namespace tinyqwen {

    // -------------------------------------------------------------------------
    // CPUBackend: CPU 计算后端
    //
    // 实现 IBackend 的所有纯虚函数，将每个算子调用转发到 dispatch 层的
    // 通用入口函数。dispatch 层根据运行时选择（--matvec-impl neon 等）
    // 决定使用哪个具体实现。
    //
    // 特点:
    //   - 零状态: 所有计算都是无状态的，不保存任何中间结果
    //   - 多线程: matvec 系列可能走多线程实现（NEON + 多线程）
    //   - 可扩展: 新添加的 CPU 优化实现通过 dispatch 注册即可生效
    // -------------------------------------------------------------------------
    class CPUBackend : public IBackend {
    public:
        CPUBackend() = default;
        ~CPUBackend() override = default;

        // =====================================================================
        // matvec 系列: 矩阵-向量乘法
        // 全部转发到 dispatch.h 的通用入口
        // =====================================================================

        // 单路 matvec: y = W @ x
        // 转发到 dispatch::matvec_f32 / matvec_f16 / matvec_i4（按 w.quant_type）
        void matvec(const WeightTensor& w, const float* x, float* y,
                   int out_dim, int in_dim) override;

        // 成对 matvec: y1 = W1 @ x, y2 = W2 @ x（共享输入 x）
        // 转发到 dispatch::matvec_pair_f32 / matvec_pair_f16 / matvec_pair_i4
        void matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                        const float* x, float* y1, float* y2,
                        int out_dim, int in_dim) override;

        // QKV 三路融合: yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        // 转发到 dispatch::matvec_qkv_f32 / matvec_qkv_f16 / matvec_qkv_i4
        void matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                       const WeightTensor& wv, const float* x,
                       float* yq, float* yk, float* yv,
                       int q_dim, int kv_dim, int in_dim) override;

        // =====================================================================
        // GEMM: 矩阵-矩阵乘法（prefill 批量处理）
        // 转发到 dispatch::matmul_f32 / matmul_i4
        // =====================================================================
        void matmul(const WeightTensor& w, const float* x, float* y,
                   int M, int K, int N) override;

        // =====================================================================
        // 非 matvec 算子: 全部走 fp32，转发到 dispatch 通用入口
        // =====================================================================

        // RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
        // 转发到 dispatch::rmsnorm
        void rmsnorm(const float* x, const float* weight, float* y,
                    int n, float eps) override;

        // 全维度 RoPE: 就地旋转 q 和 k 向量
        // 转发到 dispatch::rope
        void rope(float* q, float* k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta) override;

        // 部分 RoPE: 只旋转每个 head 的前 rotary_dim 个分量
        // 转发到 dispatch::partial_rope（Qwen3.5 用）
        void partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int rotary_dim, int pos, float theta) override;

        // decode 阶段 attention: 当前 query 与 KV cache 中所有历史计算
        // 转发到 dispatch::attention_decode
        void attention_decode(const float* q, const float* k_cache,
                             const float* v_cache, int seq_len, int max_seq_len,
                             int n_heads, int n_kv_heads, int head_dim,
                             float scale, float* out) override;

        // fp16-KV 融合 attention：直接调 kernels 的 attention_decode_f16kv_neon。
        void attention_decode_f16kv(const float* q, const uint16_t* k_cache,
                                   const uint16_t* v_cache, int seq_len, int max_seq_len,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   float scale, float* out) override;

        // SwiGLU 激活: gate[i] = silu(gate[i]) * up[i]（就地）
        // 转发到 dispatch::swiglu
        void swiglu(float* gate, const float* up, int n) override;

        // argmax: 返回最大值的下标（greedy decoding）
        // 转发到 dispatch::argmax
        int argmax(const float* logits, int n) override;

        // top-k: 取 logits 中前 k 个最大的值和下标
        // 转发到 dispatch::top_k_logits（目前直调 ref 实现）
        void top_k_logits(const float* logits, int n, int k,
                         int* indices, float* values) override;

        // =====================================================================
        // GDN 算子: Qwen3.5 线性注意力所需
        // =====================================================================

        // causal depthwise conv1d 的单步更新
        // 转发到 dispatch::causal_conv1d_update
        void causal_conv1d_update(float* x, float* state, const float* weight,
                                 float* out, int dim, int kernel_size) override;

        // L2 归一化（就地）
        // 转发到 dispatch::l2norm_inplace
        void l2norm_inplace(float* x, int n, float eps) override;

        // Gated delta rule 递归步（就地更新状态矩阵 S）
        // 转发到 dispatch::gdn_step
        void gdn_step(float* S, const float* q, const float* k, const float* v,
                     float g, float beta, float* o, int qk_dim, int v_dim) override;

        // 带门控的 RMSNorm
        // 转发到 dispatch::rmsnorm_gated
        void rmsnorm_gated(float* x, const float* gate, const float* weight,
                          float* y, int n, float eps) override;
    };

    // -------------------------------------------------------------------------
    // create_cpu_backend: 创建 CPU 后端实例的工厂函数
    //
    // 返回值:
    //   指向 IBackend 接口的 unique_ptr，实际类型为 CPUBackend
    //
    // 说明:
    //   QwenModel 通过此工厂函数获取后端，不直接依赖 CPUBackend 类型，
    //   保持接口与实现分离。
    // -------------------------------------------------------------------------
    std::unique_ptr<IBackend> create_cpu_backend();

} // namespace tinyqwen