// CUDA 后端实现：每个算子单独调用 CUDA kernel
//
// 注意：这个实现每次调用都要做 H2D/D2H 拷贝，性能不如 gpu_engine。
// 但它符合 IBackend 接口，便于 A/B 测试和单独优化每个算子。

#include "backend_cuda.h"

#ifdef TINYQWEN_HAS_CUDA

#include "../kernels/cuda/gpu_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace tinyqwen {

    namespace {
        inline void check_cuda(cudaError_t err, const char* what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
                std::abort();
            }
        }
        #define CUDA_CHECK(expr) check_cuda((expr), #expr)
    } // namespace

    CUDABackend::CUDABackend() {
        CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)));
    }

    CUDABackend::~CUDABackend() {
        if (stream_) {
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        }
        if (d_temp_) {
            cudaFree(d_temp_);
        }
    }

    void CUDABackend::ensure_temp_buffer(size_t bytes) {
        if (d_temp_size_ < bytes) {
            if (d_temp_) cudaFree(d_temp_);
            CUDA_CHECK(cudaMalloc(&d_temp_, bytes));
            d_temp_size_ = bytes;
        }
    }

    void CUDABackend::copy_to_device(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                  reinterpret_cast<cudaStream_t>(stream_)));
    }

    void CUDABackend::copy_to_host(void* dst, const void* src, size_t bytes) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                  reinterpret_cast<cudaStream_t>(stream_)));
        CUDA_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)));
    }

    // ==== matvec 系列 ====

    void CUDABackend::matvec(const WeightTensor& w, const float* x, float* y,
                            int out_dim, int in_dim) {
        // 分配 device 内存
        void* d_w;
        float* d_x;
        float* d_y;

        size_t w_bytes;
        if (w.quant_type == QuantType::kF16) {
            w_bytes = static_cast<size_t>(w.rows) * w.cols * 2;
        } else if (w.quant_type == QuantType::kI4) {
            // INT4: 每行 ceil(cols/2) bytes + scale/zero
            const int num_groups = (w.cols + w.group_size - 1) / w.group_size;
            w_bytes = static_cast<size_t>(w.rows) * ((w.cols + 1) / 2) +
                     static_cast<size_t>(w.rows) * num_groups * 2 * 2;
        } else {
            w_bytes = static_cast<size_t>(w.rows) * w.cols * 4;
        }

        CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
        CUDA_CHECK(cudaMalloc(&d_x, in_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, out_dim * sizeof(float)));

        // 拷贝数据到 device
        copy_to_device(d_w, w.data, w_bytes);
        copy_to_device(d_x, x, in_dim * sizeof(float));

        // 调用 kernel
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

        // 拷贝结果回 host
        copy_to_host(y, d_y, out_dim * sizeof(float));

        // 释放 device 内存
        cudaFree(d_w);
        cudaFree(d_x);
        cudaFree(d_y);
    }

    void CUDABackend::matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                                 const float* x, float* y1, float* y2,
                                 int out_dim, int in_dim) {
        // TODO: 实现 pair matvec
        // 暂时回退到两次单独调用
        matvec(w1, x, y1, out_dim, in_dim);
        matvec(w2, x, y2, out_dim, in_dim);
    }

    void CUDABackend::matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                                const WeightTensor& wv, const float* x,
                                float* yq, float* yk, float* yv,
                                int q_dim, int kv_dim, int in_dim) {
        // TODO: 实现 qkv matvec
        // 暂时回退到三次单独调用
        matvec(wq, x, yq, q_dim, in_dim);
        matvec(wk, x, yk, kv_dim, in_dim);
        matvec(wv, x, yv, kv_dim, in_dim);
    }

    void CUDABackend::matmul(const WeightTensor& w, const float* x, float* y,
                            int M, int K, int N) {
        // TODO: 实现 GEMM
        // 暂时回退到 N 次 matvec
        for (int c = 0; c < N; ++c) {
            matvec(w, x + static_cast<size_t>(c) * K,
                  y + static_cast<size_t>(c) * M, M, K);
        }
    }

    // ==== 非 matvec 算子 ====

    void CUDABackend::rmsnorm(const float* x, const float* weight, float* y,
                             int n, float eps) {
        float* d_x;
        float* d_w;
        float* d_y;

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

    void CUDABackend::rope(float* q, float* k, int n_heads, int n_kv_heads,
                          int head_dim, int pos, float theta) {
        const int q_dim = n_heads * head_dim;
        const int kv_dim = n_kv_heads * head_dim;

        float* d_q;
        float* d_k;
        int* d_pos;

        CUDA_CHECK(cudaMalloc(&d_q, q_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k, kv_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pos, sizeof(int)));

        copy_to_device(d_q, q, q_dim * sizeof(float));
        copy_to_device(d_k, k, kv_dim * sizeof(float));
        copy_to_device(d_pos, &pos, sizeof(int));

        gpu::rope(reinterpret_cast<cudaStream_t>(stream_), d_q, d_k, n_heads, n_kv_heads,
                 head_dim, d_pos, theta);

        copy_to_host(q, d_q, q_dim * sizeof(float));
        copy_to_host(k, d_k, kv_dim * sizeof(float));

        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_pos);
    }

    void CUDABackend::partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                  int head_dim, int rotary_dim, int pos, float theta) {
        // TODO: 实现 partial_rope CUDA kernel
        // 暂时回退到 CPU
        std::fprintf(stderr, "CUDABackend: partial_rope not yet implemented\n");
        std::abort();
    }

    void CUDABackend::attention_decode(const float* q, const float* k_cache,
                                      const float* v_cache, int seq_len, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      float scale, float* out) {
        const int q_dim = n_heads * head_dim;
        const int kv_dim = n_kv_heads * head_dim;
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

        const int pos = seq_len - 1;
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

    void CUDABackend::swiglu(float* gate, const float* up, int n) {
        float* d_gate;
        float* d_up;

        CUDA_CHECK(cudaMalloc(&d_gate, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up, n * sizeof(float)));

        copy_to_device(d_gate, gate, n * sizeof(float));
        copy_to_device(d_up, up, n * sizeof(float));

        gpu::swiglu(reinterpret_cast<cudaStream_t>(stream_), d_gate, d_up, n);

        copy_to_host(gate, d_gate, n * sizeof(float));

        cudaFree(d_gate);
        cudaFree(d_up);
    }

    int CUDABackend::argmax(const float* logits, int n) {
        float* d_logits;
        int* d_out;

        CUDA_CHECK(cudaMalloc(&d_logits, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));

        copy_to_device(d_logits, logits, n * sizeof(float));

        gpu::argmax(reinterpret_cast<cudaStream_t>(stream_), d_logits, d_out, n);

        int result;
        copy_to_host(&result, d_out, sizeof(int));

        cudaFree(d_logits);
        cudaFree(d_out);

        return result;
    }

    void CUDABackend::top_k_logits(const float* logits, int n, int k,
                                  int* indices, float* values) {
        // TODO: 实现 top_k CUDA kernel
        // 暂时回退到 CPU
        std::fprintf(stderr, "CUDABackend: top_k_logits not yet implemented\n");
        std::abort();
    }

    // ==== GDN 算子 ====

    void CUDABackend::causal_conv1d_update(float* x, float* state, const float* weight,
                                          float* out, int dim, int kernel_size) {
        // TODO: 实现 causal_conv1d_update CUDA kernel
        std::fprintf(stderr, "CUDABackend: causal_conv1d_update not yet implemented\n");
        std::abort();
    }

    void CUDABackend::l2norm_inplace(float* x, int n, float eps) {
        // TODO: 实现 l2norm_inplace CUDA kernel
        std::fprintf(stderr, "CUDABackend: l2norm_inplace not yet implemented\n");
        std::abort();
    }

    void CUDABackend::gdn_step(float* S, const float* q, const float* k, const float* v,
                              float g, float beta, float* o, int qk_dim, int v_dim) {
        // TODO: 实现 gdn_step CUDA kernel
        std::fprintf(stderr, "CUDABackend: gdn_step not yet implemented\n");
        std::abort();
    }

    void CUDABackend::rmsnorm_gated(float* x, const float* gate, const float* weight,
                                   float* y, int n, float eps) {
        // TODO: 实现 rmsnorm_gated CUDA kernel
        std::fprintf(stderr, "CUDABackend: rmsnorm_gated not yet implemented\n");
        std::abort();
    }

    std::unique_ptr<IBackend> create_cuda_backend() {
        return std::make_unique<CUDABackend>();
    }

} // namespace tinyqwen

#endif // TINYQWEN_HAS_CUDA
