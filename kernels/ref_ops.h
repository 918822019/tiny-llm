#pragma once

// tinyqwen runtime 的朴素标量 reference kernels。
//
// 本目录下所有 kernel 的共同约定：
//   - 正确性和可读性优先，不用 SIMD、不开多线程；
//   - 输入/输出指针由调用方提供，kernel 不为其分配内存；
//   - shape 全部以显式参数传入，kernel 无隐藏状态；
//   - 每个 kernel 在 tests/ 都有小 shape 单元测试。

namespace tinyqwen {

// RMSNorm（Qwen 定义，只有 weight）：y = x / sqrt(mean(x^2) + eps) * weight
void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps);

// y = W @ x；W 为行主序 [out_dim, in_dim]（HF 布局，不做转置）。
void matvec_f32_ref(const float* w, const float* x, float* y, int out_dim, int in_dim);

// 数值稳定 softmax：y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x))
void softmax_ref(const float* x, float* y, int n);

// SiLU：y = x * sigmoid(x)
void silu_ref(const float* x, float* y, int n);

// 返回第一个最大值的下标（平局取靠前者）。
int argmax_ref(const float* logits, int n);

// 对单个 decode 位置做旋转位置编码（in-place）。
// q: [n_heads * head_dim]，k: [n_kv_heads * head_dim]。
// 与 HF Qwen2 的 rotate-half 约定一致：
//   out[i]        = x[i] * cos - x[i + half] * sin
//   out[i + half] = x[i + half] * cos + x[i] * sin
// 其中 inv_freq[i] = theta ^ (-2i / head_dim)，angle = pos * inv_freq[i]。
void rope_ref(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta);

// decode 阶段（单个 query 位置）对 KV cache 做 attention。
//   q       : [n_heads * head_dim]
//   k_cache : [n_kv_heads][max_seq_len][head_dim]，有效位置为 [0, seq_len)
//   v_cache : 同上布局
//   out     : [n_heads * head_dim]
// score = dot(q_h, k_{kv(h), t}) * scale；采用 online softmax，无暂存内存。
// GQA 映射：query head h 对应的 kv head 为 h / (n_heads / n_kv_heads)。
void attention_decode_ref(const float* q, const float* k_cache, const float* v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float* out);

}  // namespace tinyqwen
