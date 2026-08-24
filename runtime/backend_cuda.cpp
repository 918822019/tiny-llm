// ============================================================================
// backend_cuda.cpp — CUDA 后端实现
// ============================================================================
// 本文件实现 CUDABackend 类，它是推理引擎的 GPU 计算后端，实现了 IBackend
// 接口。每个算子单独调用 CUDA kernel，通过 gpu_kernels.cuh 中定义的 GPU 函数。
//
// 注意：这个实现每次调用都要做 Host-to-Device (H2D) 和 Device-to-Host (D2H)
// 拷贝，因此性能不如 gpu_engine（后者将所有权重和激活常驻在 GPU 显存中）。
// 但它的优势在于：
//   1. 符合 IBackend 接口，CPU 和 GPU 代码路径统一
//   2. 便于 A/B 测试，单独优化每个算子
//   3. 作为 gpu_engine 的参考实现和正确性基准
//
// 当前实现状态：
//   - f32/f16 matvec: 已实现
//   - INT4 matvec: 未实现（TODO）
//   - matvec_pair/matvec_qkv: 回退到多次单独调用
//   - rmsnorm/rope/attention_decode/swiglu/argmax: 已实现
//   - GDN 系列算子（causal_conv1d_update/l2norm/gdn_step/rmsnorm_gated）: 未实现
// ============================================================================

#include "backend_cuda.h"

#ifdef TINYQWEN_HAS_CUDA

#include "../kernels/cuda/gpu_kernels.cuh"

#include <cstdio>       // 标准输入输出
#include <cstdlib>      // 标准库（abort）
#include <cstring>      // 内存操作
#include <cuda_runtime.h> // CUDA 运行时 API

namespace tinyqwen {

    namespace {
        // =====================================================================
        // check_cuda() — CUDA 错误检查辅助函数
        // =====================================================================
        // 参数：
        //   err  — CUDA API 返回的错误码
        //   what — 操作描述字符串（用于错误信息）
        // 说明：如果 err 不是 cudaSuccess，打印错误信息并 abort。
        //       这是典型的 fail-fast 策略：CUDA 错误通常不可恢复。
        inline void check_cuda(cudaError_t err, const char* what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
                std::abort();
            }
        }
        // CUDA_CHECK 宏：自动传入操作表达式作为描述
        #define CUDA_CHECK(expr) check_cuda((expr), #expr)
    } // namespace

    // =========================================================================
    // CUDABackend::CUDABackend() — 构造函数
    // =========================================================================
    // 说明：创建 CUDA stream，所有后续操作将在此 stream 上异步执行。
    CUDABackend::CUDABackend() {
        CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)));
    }

    // =========================================================================
    // CUDABackend::~CUDABackend() — 析构函数
    // =========================================================================
    // 说明：销毁 CUDA stream 和临时缓冲区，释放 GPU 资源。
    CUDABackend::~CUDABackend() {
        if (stream_) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        }
        if (d_temp_) {
            cudaFree(d_temp_); // 释放临时 GPU 内存
        }
    }

    // =========================================================================
    // CUDABackend::ensure_temp_buffer() — 确保临时缓冲区足够大
    // =========================================================================
    // 参数：
    //   bytes — 所需的最小字节数
    // 说明：如果当前临时缓冲区小于所需大小，释放旧的并重新分配。
    void CUDABackend::ensure_temp_buffer(size_t bytes) {
        if (d_temp_size_ < bytes) {
            if (d_temp_) cudaFree(d_temp_);
            CUDA_CHECK(cudaMalloc(&d_temp_, bytes));
            d_temp_size_ = bytes;
        }
    }

    // =========================================================================
    // CUDABackend::copy_to_device() — Host-to-Device 数据拷贝
    // =========================================================================
    // 参数：
    //   dst   — GPU 目标地址
    //   src   — CPU 源地址
    //   bytes — 拷贝字节数
    // 说明：异步拷贝，使用 cudaMemcpyAsync 在 CUDA stream 上执行。
    void CUDABackend::copy_to_device(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                  reinterpret_cast<cudaStream_t>(stream_)));
    }

    // =========================================================================
    // CUDABackend::copy_to_host() — Device-to-Host 数据拷贝
    // =========================================================================
    // 参数：
    //   dst   — CPU 目标地址
    //   src   — GPU 源地址
    //   bytes — 拷贝字节数
    // 说明：异步拷贝后同步等待完成，确保数据已传回 CPU。
    void CUDABackend::copy_to_host(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                  reinterpret_cast<cudaStream_t>(stream_)));
        CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)));
    }

    // ==== matvec 系列 ====

    // 矩阵-向量乘法：分配 GPU 内存 -> 拷贝数据 -> 调用 kernel -> 拷贝回 CPU
    void CUDABackend::matvec(const WeightTensor& w, const float* x, float* y,
                            int out_dim, int in_dim) {
        void* d_w;    // GPU 上的权重
        float* d_x;   // GPU 上的输入向量
        float* d_y;   // GPU 上的输出向量

        // 计算权重在 GPU 上需要的字节数（根据量化类型）
        size_t w_bytes;
        if (w.quant_type == QuantType::kF16) {
            w_bytes = static_cast<size_t>(w.rows) * w.cols * 2; // 每元素 2 字节
        } else if (w.quant_type == QuantType::kI4) {
            // INT4: 每行 ceil(cols/2) bytes + scale/zero（每组 2 float + 2 float）
            const int num_groups = (w.cols + w.group_size - 1) / w.group_size;
            w_bytes = static_cast<size_t>(w.rows) * ((w.cols + 1) / 2) +
                     static_cast<size_t>(w.rows) * num_groups * 2 * 2;
        } else {
            w_bytes = static_cast<size_t>(w.rows) * w.cols * 4; // 每元素 4 字节
        }

        // 分配 GPU 内存
        CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
        CUDA_CHECK(cudaMalloc(&d_x, in_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, out_dim * sizeof(float)));

        // 拷贝数据到 GPU
        copy_to_device(d_w, w.data, w_bytes);
        copy_to_device(d_x, x, in_dim * sizeof(float));

        // 调用 CUDA kernel
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
        if (w.quant_type == QuantType::kF16) {
            gpu::matvec_f16(stream, static_cast<const uint16_t*>(d_w), d_x, d_y, out_dim, in_dim);
        } else if (w.quant_type == QuantType::kI4) {
            // TODO: 实现 INT4 CUDA kernel
            std::fprintf(stderr, "CUDABackend: INT4 not yet implemented\n");
            std::abort();
        } else {
            gpu::matvec_f32(stream, static_cast<const float*>(d_w), d_x, d_y, out_dim, in_dim);
        }

        // 拷贝结果回 CPU
        copy_to_host(y, d_y, out_dim * sizeof(float));

        // 释放 GPU 内存
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    // 双输出 matvec：TODO，当前回退到两次单独调用
    void CUDABackend::matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                                 const float* x, float* y1, float* y2,
                                 int out_dim, int in_dim) {
        // TODO: 实现 pair matvec（融合版本可减少 H2D 拷贝）
        matvec(w1, x, y1, out_dim, in_dim);
        matvec(w2, x, y2, out_dim, in_dim);
    }

    // 三输出 QKV matvec：TODO，当前回退到三次单独调用
    void CUDABackend::matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                                const WeightTensor& wv, const float* x,
                                float* yq, float* yk, float* yv,
                                int q_dim, int kv_dim, int in_dim) {
        // TODO: 实现 qkv matvec（融合版本可减少 H2D 拷贝）
        matvec(wq, x, yq, q_dim, in_dim);
        matvec(wk, x, yk, kv_dim, in_dim);
        matvec(wv, x, yv, kv_dim, in_dim);
    }

    // 批量矩阵乘法：TODO，当前回退到 N 次 matvec
    void CUDABackend::matmul(const WeightTensor& w, const float* x, float* y,
                            int M, int K, int N) {
        // TODO: 实现 GEMM（使用 cuBLAS 或手写 kernel，效率远高于 N 次 matvec）
        for (int c = 0; c < N; ++c) {
            matvec(w, x + static_cast<size_t>(c) * K,
                  y + static_cast<size_t>(c) * M, M, K);
        }
    }

    // ==== 非 matvec 算子 ====

    // RMS 归一化：分配 GPU 内存 -> 拷贝 -> kernel -> 拷贝回 CPU
    void CUDABackend::rmsnorm(const float* x, const float* weight, float* y,
                             int n, float eps) {
        float* d_x;  // GPU 输入
        float* d_w;  // GPU 权重
        float* d_y;  // GPU 输出

        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_w, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));

        copy_to_device(d_x, x, n * sizeof(float));
        copy_to_device(d_w, weight, n * sizeof(float));

        gpu::rmsnorm(reinterpret_cast<cudaStream_t>(stream_), d_y, d_x, d_w, n, eps);

        copy_to_host(y, d_y, n * sizeof(float));

        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_y);
    }

    // 旋转位置编码（全头 RoPE）
    void CUDABackend::rope(float* q, float* k, int n_heads, int n_kv_heads,
                          int head_dim, int pos, float theta) {
        const int q_dim = n_heads * head_dim;
        const int kv_dim = n_kv_heads * head_dim;

        float* d_q;
        float* d_k;
        int* d_pos; // 位置参数，在 GPU 上作为标量

        CUDA_CHECK(cudaMalloc(&d_q, q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));

        copy_to_device(d_q, q, q_dim * sizeof(float));
        copy_to_device(d_k, k, kv_dim * sizeof(float));
        copy_to_device(d_pos, &pos, sizeof(int));

        gpu::rope(reinterpret_cast<cudaStream_t>(stream_), d_q, d_k, n_heads, n_kv_heads,
                 head_dim, d_pos, theta);

        // 原地修改：需要将结果拷回原始数组
        copy_to_host(q, d_q, q_dim * sizeof(float));
        copy_to_host(k, d_k, kv_dim * sizeof(float));

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_pos);
    }

    // 部分旋转位置编码：TODO，未实现
    void CUDABackend::partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                  int head_dim, int rotary_dim, int pos, float theta) {
        // TODO: 实现 partial_rope CUDA kernel
        std::fprintf(stderr, "CUDABackend: partial_rope not yet implemented\n");
        std::abort();
    }

    // 单步 attention 解码
    void CUDABackend::attention_decode(const float* q, const float* k_cache,
                                      const float* v_cache, int seq_len, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      float scale, float* out) {
        const int q_dim = n_heads * head_dim;
        const int kv_cache_size = n_kv_heads * max_seq_len * head_dim;

        float* d_q;
        float* d_k_cache;
        float* d_v_cache;
        float* d_out;
        int* d_pos;

        CUDA_CHECK(cudaMalloc(&d_q, q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_cache, kv_cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache, kv_cache_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));

        copy_to_device(d_q, q, q_dim * sizeof(float));
        copy_to_device(d_k_cache, k_cache, kv_cache_size * sizeof(float));
        copy_to_device(d_v_cache, v_cache, kv_cache_size * sizeof(float));

        const int pos = seq_len - 1; // 当前 token 的位置
        copy_to_device(d_pos, &pos, sizeof(int));

        gpu::attention_decode(reinterpret_cast<cudaStream_t>(stream_), d_out, d_q,
                            d_k_cache, d_v_cache, d_pos, max_seq_len, n_heads, n_kv_heads,
                            head_dim, scale);

        copy_to_host(out, d_out, q_dim * sizeof(float));

        cudaFree(d_q);
        cudaFree(d_k_cache);
        cudaFree(d_v_cache);
        cudaFree(d_out);
        cudaFree(d_pos);
    }

    // fp16-KV 融合 attention：CUDA 后端暂不支持，调用即 abort。
    void CUDABackend::attention_decode_f16kv(const float* q, const uint16_t* k_cache,
                                            const uint16_t* v_cache, int seq_len, int max_seq_len,
                                            int n_heads, int n_kv_heads, int head_dim,
                                            float scale, float* out) {
        (void)q; (void)k_cache; (void)v_cache; (void)seq_len; (void)max_seq_len;
        (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)scale; (void)out;
        std::fprintf(stderr, "tinyqwen: CUDABackend::attention_decode_f16kv 未实现（fp16-KV 仅 CPU）\n");
        std::abort();
    }

    // SwiGLU 激活函数
    void CUDABackend::swiglu(float* gate, const float* up, int n) {
        float* d_gate;
        float* d_up;

        CUDA_CHECK(cudaMalloc(&d_gate, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up, n * sizeof(float)));

        copy_to_device(d_gate, gate, n * sizeof(float));
        copy_to_device(d_up, up, n * sizeof(float));

        gpu::swiglu(reinterpret_cast<cudaStream_t>(stream_), d_gate, d_up, n);

        copy_to_host(gate, d_gate, n * sizeof(float)); // 原地修改结果

        cudaFree(d_gate);
        cudaFree(d_up);
    }

    // 取最大值索引
    int CUDABackend::argmax(const float* logits, int n) {
        float* d_logits;
        int* d_out; // GPU 上的结果（单个 int）

        CUDA_CHECK(cudaMalloc(&d_logits, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

        copy_to_device(d_logits, logits, n * sizeof(float));

        gpu::argmax(reinterpret_cast<cudaStream_t>(stream_), d_logits, d_out, n);

        int result;
        copy_to_host(&result, d_out, sizeof(int)); // 拷贝单个 int 结果

        cudaFree(d_logits);
        cudaFree(d_out);

        return result;
    }

    // top-k logits：TODO，未实现
    void CUDABackend::top_k_logits(const float* logits, int n, int k,
                                  int* indices, float* values) {
        // TODO: 实现 top_k CUDA kernel
        std::fprintf(stderr, "CUDABackend: top_k_logits not yet implemented\n");
        std::abort();
    }

    // ==== GDN 算子（全部未实现）====

    void CUDABackend::causal_conv1d_update(float* x, float* state, const float* weight,
                                          float* out, int dim, int kernel_size) {
        std::fprintf(stderr, "CUDABackend: causal_conv1d_update not yet implemented\n");
        std::abort();
    }

    void CUDABackend::l2norm_inplace(float* x, int n, float eps) {
        std::fprintf(stderr, "CUDABackend: l2norm_inplace not yet implemented\n");
        std::abort();
    }

    void CUDABackend::gdn_step(float* S, const float* q, const float* k, const float* v,
                              float g, float beta, float* o, int qk_dim, int v_dim) {
        std::fprintf(stderr, "CUDABackend: gdn_step not yet implemented\n");
        std::abort();
    }

    void CUDABackend::rmsnorm_gated(float* x, const float* gate, const float* weight,
                                   float* y, int n, float eps) {
        std::fprintf(stderr, "CUDABackend: rmsnorm_gated not yet implemented\n");
        std::abort();
    }

    // CUDA 后端工厂函数
    std::unique_ptr<IBackend> create_cuda_backend() {
        return std::make_unique<CUDABackend>();
    }

} // namespace tinyqwen

#endif // TINYQWEN_HAS_CUDA