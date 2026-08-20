#pragma once

// CUDA 后端实现：每个算子单独调用 CUDA kernel
//
// 与 gpu_engine.cu 的区别：
//   - gpu_engine：整段 forward 一起跑（性能最好，但不符合 IBackend 接口）
//   - backend_cuda：每个算子单独调用（性能略差，但符合 IBackend 接口，便于 A/B 测试）

#include "backend.h"
#include <memory>

#ifdef TINYQWEN_HAS_CUDA

namespace tinyqwen {

    class CUDABackend : public IBackend {
    public:
        CUDABackend();
        ~CUDABackend() override;

        // matvec 系列
        void matvec(const WeightTensor& w, const float* x, float* y,
                   int out_dim, int in_dim) override;

        void matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                        const float* x, float* y1, float* y2,
                        int out_dim, int in_dim) override;

        void matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                       const WeightTensor& wv, const float* x,
                       float* yq, float* yk, float* yv,
                       int q_dim, int kv_dim, int in_dim) override;

        // GEMM
        void matmul(const WeightTensor& w, const float* x, float* y,
                   int M, int K, int N) override;

        // 非 matvec 算子
        void rmsnorm(const float* x, const float* weight, float* y,
                    int n, float eps) override;

        void rope(float* q, float* k, int n_heads, int n_kv_heads,
                 int head_dim, int pos, float theta) override;

        void partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int rotary_dim, int pos, float theta) override;

        void attention_decode(const float* q, const float* k_cache,
                             const float* v_cache, int seq_len, int max_seq_len,
                             int n_heads, int n_kv_heads, int head_dim,
                             float scale, float* out) override;

        void swiglu(float* gate, const float* up, int n) override;

        int argmax(const float* logits, int n) override;

        void top_k_logits(const float* logits, int n, int k,
                         int* indices, float* values) override;

        // GDN 算子
        void causal_conv1d_update(float* x, float* state, const float* weight,
                                 float* out, int dim, int kernel_size) override;

        void l2norm_inplace(float* x, int n, float eps) override;

        void gdn_step(float* S, const float* q, const float* k, const float* v,
                     float g, float beta, float* o, int qk_dim, int v_dim) override;

        void rmsnorm_gated(float* x, const float* gate, const float* weight,
                          float* y, int n, float eps) override;

    private:
        // CUDA stream（所有 kernel 在同一个 stream 上串行）
        void* stream_ = nullptr;  // cudaStream_t

        // 临时 buffer（避免每个算子都分配显存）
        void* d_temp_ = nullptr;
        size_t d_temp_size_ = 0;

        // 确保临时 buffer 足够大
        void ensure_temp_buffer(size_t bytes);

        // 从 host 拷贝到 device
        void copy_to_device(void* dst, const void* src, size_t bytes);

        // 从 device 拷贝到 host
        void copy_to_host(void* dst, const void* src, size_t bytes);
    };

    // 创建 CUDA 后端实例
    std::unique_ptr<IBackend> create_cuda_backend();

} // namespace tinyqwen

#endif // TINYQWEN_HAS_CUDA
