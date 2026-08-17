#pragma once

// GPU decode engine 的 kernel 启动器接口（device 指针版）。
//
// 这一层把 forward 需要的每个算子都做成"接收 device 指针、在指定 stream 上
// 启动"的 host 启动器，供 gpu_engine.cu 的 GpuDecodeEngine 编排，也供
// tests/test_gpu_ops.cu 逐个对齐 *_ref。所有数据指针都是 **device 指针**
// （logits/idx 也是），函数内部不做任何 H2D/D2H——搬运由 engine 统一管。
//
// 与 kernels/ 里按算子分发的变体不同：这些启动器**不**注册进 dispatch，是
// engine 的内部构件。matvec 的 coalesced kernel 在这里自带一份（与
// matvec_f*_cuda_resident_coal*.cu 同款算法，device 指针形态），保持 engine
// 自包含、不去改已测好的分发变体。
//
// 仅 CUDA 构建编译（见 kernels/CMakeLists.txt 的 TINYQWEN_HAS_CUDA gate）。

#include <cstdint>
#include <cuda_runtime.h>

namespace tinyqwen {
namespace gpu {

    // ---- 非 matvec 算子 ----
    // hidden[token] = embed[token*hidden_dim ..]，f16 时逐元素转 fp32。
    // token 传 device 指针（而非按值传参）：CUDA Graph capture 一次、多次
    // replay 时，每步变化的标量必须来自固定地址，kernel 内解引用读取当前值。
    void embed_lookup(cudaStream_t s, float *hidden, const void *embed, const int *token,
                      int hidden_dim, bool is_f16);
    // y = x / sqrt(mean(x^2)+eps) * w（对齐 rmsnorm_ref）。
    void rmsnorm(cudaStream_t s, float *y, const float *x, const float *w, int n, float eps);
    // x[i] += bias[i]（q/k/v 投影后的 bias）。
    void bias_add(cudaStream_t s, float *x, const float *bias, int n);
    // hidden[i] += delta[i]（残差连接）。
    void residual_add(cudaStream_t s, float *hidden, const float *delta, int n);
    // RoPE in-place（对齐 rope_ref，rotate-half）。pos 传 device 指针，理由同 embed_lookup。
    void rope(cudaStream_t s, float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
              const int *pos, float theta);
    // 把 k_in/v_in 追加进某层 KV plane 的 pos 槽位（对齐 qwen_model 的 memcpy 索引）。
    // pos 传 device 指针，理由同 embed_lookup。
    void kv_append(cudaStream_t s, float *k_plane, float *v_plane, const float *k_in,
                   const float *v_in, const int *pos, int n_kv_heads, int max_seq_len,
                   int head_dim);
    // decode attention，online softmax + GQA（对齐 attention_decode_ref）。
    // pos 传 device 指针（seq_len = *pos + 1，在 kernel 内算），理由同 embed_lookup。
    void attention_decode(cudaStream_t s, float *out, const float *q, const float *k_cache,
                          const float *v_cache, const int *pos, int max_seq_len, int n_heads,
                          int n_kv_heads, int head_dim, float scale);
    // gate[i] = silu(gate[i]) * up[i]（in-place，对齐 swiglu_ref）。
    void swiglu(cudaStream_t s, float *gate, const float *up, int n);
    // 全词表 argmax（严格 > 取首个，对齐 argmax_ref），结果写 device 的 out_idx。
    void argmax(cudaStream_t s, const float *logits, int *out_idx, int n);

    // ---- matvec（coalesced，device 指针）----
    void matvec_f16(cudaStream_t s, const uint16_t *w, const float *x, float *y, int out_dim,
                    int in_dim);
    void matvec_f16_pair(cudaStream_t s, const uint16_t *w1, const uint16_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim);
    void matvec_f16_qkv(cudaStream_t s, const uint16_t *wq, const uint16_t *wk,
                        const uint16_t *wv, const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim);
    void matvec_f32(cudaStream_t s, const float *w, const float *x, float *y, int out_dim,
                    int in_dim);
    void matvec_f32_pair(cudaStream_t s, const float *w1, const float *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim);
    void matvec_f32_qkv(cudaStream_t s, const float *wq, const float *wk, const float *wv,
                        const float *x, float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                        int in_dim);

} // namespace gpu
} // namespace tinyqwen
