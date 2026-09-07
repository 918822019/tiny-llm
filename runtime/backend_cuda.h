#pragma once

// ============================================================================
// 文件: backend_cuda.h
// 作用: CUDA 后端实现 —— 每个算子单独调用 CUDA kernel
//
// 与 gpu_engine.cu 的区别:
//   - gpu_engine: 整段 forward 一起跑（性能最好，但不符合 IBackend 接口）
//     所有权重/激活/KV cache 全部常驻显存，整段 forward 在 GPU 上完成，
//     只有 token id 和 logits 过 PCIe 总线
//   - backend_cuda: 每个算子单独调用（性能略差，但符合 IBackend 接口，
//     便于 A/B 测试和性能调试）
//     每个算子独立调用 CUDA kernel，中间结果可能需要在 CPU/GPU 之间传输
//
// 为什么需要两种 CUDA 实现?
//   - gpu_engine: 追求极致性能的生产路径
//   - backend_cuda: 用于逐算子性能分析和与 CPU 实现的逐位对比验证
//     通过 IBackend 接口，可以无缝切换 CPU/CUDA，验证数值正确性
//
// 编译条件:
//   仅在定义了 TINYQWEN_HAS_CUDA 宏时可用（由 CMake 根据 CUDA 工具链决定）
// ============================================================================

#include "backend.h"
#include <memory>

#ifdef TINYQWEN_HAS_CUDA

namespace tinyqwen {

    // -------------------------------------------------------------------------
    // CUDABackend: CUDA 计算后端
    //
    // 实现 IBackend 的所有纯虚函数，每个算子独立调用 CUDA kernel。
    //
    // 内部管理:
    //   - CUDA stream: 所有 kernel 在同一个 stream 上串行执行，确保顺序正确
    //   - 临时 buffer: 避免每个算子都分配/释放显存，复用一块 d_temp_ buffer
    //   - 数据拷贝: 提供 copy_to_device / copy_to_host 辅助函数
    //
    // 使用限制:
    //   - 需要有 CUDA 兼容的 GPU 和 CUDA 运行时库
    //   - 仅在编译时定义了 TINYQWEN_HAS_CUDA 时可用
    // -------------------------------------------------------------------------
    class CUDABackend : public IBackend {
    public:
        // 构造函数: 初始化 CUDA 资源和 stream
        CUDABackend();

        // 析构函数: 释放 CUDA 资源和临时 buffer
        ~CUDABackend() override;

        // =====================================================================
        // matvec 系列: 矩阵-向量乘法
        // 每个调用启动对应的 CUDA kernel
        // =====================================================================

        // 单路 matvec: y = W @ x
        // W 在 GPU 显存中（提前拷贝），x 和 y 在 host 或 device 内存中
        void matvec(const WeightTensor& w, const float* x, float* y,
                   int out_dim, int in_dim) override;

        // 成对 matvec: y1 = W1 @ x, y2 = W2 @ x（共享输入 x）
        // 两个 matvec 在同一个 kernel launch 中完成
        void matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                        const float* x, float* y1, float* y2,
                        int out_dim, int in_dim) override;

        // QKV 三路融合: yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        // 三个 matvec 在同一个 kernel launch 中完成
        void matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                       const WeightTensor& wv, const float* x,
                       float* yq, float* yk, float* yv,
                       int q_dim, int kv_dim, int in_dim) override;

        // =====================================================================
        // GEMM: 矩阵-矩阵乘法（prefill 批量处理）
        // 使用 cuBLAS 或自定义 CUDA GEMM kernel
        // =====================================================================
        void matmul(const WeightTensor& w, const float* x, float* y,
                   int M, int K, int N) override;

        // =====================================================================
        // 非 matvec 算子: 全部走 fp32
        // 每个调用启动对应的 CUDA kernel
        // =====================================================================

        // RMSNorm: 使用 CUDA reduction 计算均方根
        void rmsnorm(const float* x, const float* weight, float* y,
                    int n, float eps) override;

        // 全维度 RoPE: 就地旋转 q 和 k 向量
        void rope(float* q, float* k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta) override;

        // 部分 RoPE: 只旋转每个 head 的前 rotary_dim 个分量
        void partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int rotary_dim, int pos, float theta) override;

        // decode attention: 使用 flash attention 风格的 CUDA kernel
        void attention_decode(const float* q, const float* k_cache,
                             const float* v_cache, int seq_len, int max_seq_len,
                             int n_heads, int n_kv_heads, int head_dim,
                             float scale, float* out) override;

        // fp16-KV 融合 attention：CUDA 后端暂不支持 fp16-KV，调用即 abort。
        void attention_decode_f16kv(const float* q, const uint16_t* k_cache,
                                   const uint16_t* v_cache, int seq_len, int max_seq_len,
                                   int n_heads, int n_kv_heads, int head_dim,
                                   float scale, float* out) override;

        // SwiGLU 激活: 使用 CUDA element-wise kernel
        void swiglu(float* gate, const float* up, int n) override;

        // argmax: 使用 CUDA reduction 找最大值下标
        int argmax(const float* logits, int n) override;

        // top-k: 使用 CUDA 的 top-k 算法
        void top_k_logits(const float* logits, int n, int k,
                         int* indices, float* values) override;

        // MoE 路由 top-k + softmax：CUDA 后端暂未实现，调用即 abort。
        void topk_softmax(const float* gate_logits, int n, int k,
                          int* indices, float* weights) override;

        // =====================================================================
        // GDN 算子: Qwen3.5 线性注意力所需
        // =====================================================================

        // causal conv1d 单步更新
        void causal_conv1d_update(float* x, float* state, const float* weight,
                                 float* out, int dim, int kernel_size) override;

        // L2 归一化（就地）
        void l2norm_inplace(float* x, int n, float eps) override;

        // Gated delta rule 递归步
        void gdn_step(float* S, const float* q, const float* k, const float* v,
                     float g, float beta, float* o, int qk_dim, int v_dim) override;

        // 带门控的 RMSNorm
        void rmsnorm_gated(float* x, const float* gate, const float* weight,
                          float* y, int n, float eps) override;

    private:
        // ---------------------------------------------------------------------
        // stream_: CUDA stream 句柄（cudaStream_t）
        //
        // 所有 kernel 在同一个 stream 上串行执行，保证:
        //   - 前一个 kernel 完成后再启动下一个（保证计算顺序）
        //   - 不需要额外的同步原语
        //   - 与 host 的异步执行可以重叠（copy 和 compute 同时进行）
        // ---------------------------------------------------------------------
        void* stream_ = nullptr;  // 实际类型为 cudaStream_t，用 void* 避免头文件引入 CUDA

        // ---------------------------------------------------------------------
        // d_temp_ / d_temp_size_: 临时 GPU buffer
        //
        // 复用一块 GPU 显存作为临时缓冲区，避免每个算子都分配/释放显存。
        // 当需要更大空间时，ensure_temp_buffer 会释放旧 buffer 并分配新 buffer。
        // ---------------------------------------------------------------------
        void* d_temp_ = nullptr;   // 实际类型为 void*，指向 GPU 显存
        size_t d_temp_size_ = 0;   // 当前 buffer 的大小（字节）

        // ---------------------------------------------------------------------
        // ensure_temp_buffer: 确保临时 buffer 至少为 bytes 字节
        //
        // 参数:
        //   bytes: 需要的最小字节数
        //
        // 说明:
        //   如果当前 buffer 不够大，释放旧 buffer 并分配新 buffer。
        //   如果已经够大，不做任何操作（避免不必要的显存分配）。
        // ---------------------------------------------------------------------
        void ensure_temp_buffer(size_t bytes);

        // ---------------------------------------------------------------------
        // copy_to_device: 从 host 内存拷贝到 device 显存
        //
        // 参数:
        //   dst:    device 目标地址
        //   src:    host 源地址
        //   bytes:  拷贝字节数
        //
        // 说明:
        //   使用 cudaMemcpyAsync 在 stream_ 上异步拷贝，
        //   与 kernel 执行可以重叠。
        // ---------------------------------------------------------------------
        void copy_to_device(void* dst, const void* src, size_t bytes);

        // ---------------------------------------------------------------------
        // copy_to_host: 从 device 显存拷贝到 host 内存
        //
        // 参数:
        //   dst:    host 目标地址
        //   src:    device 源地址
        //   bytes:  拷贝字节数
        // ---------------------------------------------------------------------
        void copy_to_host(void* dst, const void* src, size_t bytes);
    };

    // -------------------------------------------------------------------------
    // create_cuda_backend: 创建 CUDA 后端实例的工厂函数
    //
    // 返回值:
    //   指向 IBackend 接口的 unique_ptr，实际类型为 CUDABackend
    //
    // 说明:
    //   仅在 TINYQWEN_HAS_CUDA 定义时可用。
    //   如果 CUDA 初始化失败（无 GPU 等），返回 nullptr。
    // -------------------------------------------------------------------------
    std::unique_ptr<IBackend> create_cuda_backend();

} // namespace tinyqwen

#endif // TINYQWEN_HAS_CUDA