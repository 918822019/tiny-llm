#pragma once

// Naive scalar reference kernels for the tinyqwen runtime.
//
// Rules for every kernel in this directory:
//   - correctness and readability first, NO SIMD, no threading;
//   - input/output pointers are caller-owned; kernels never allocate for them;
//   - shapes are explicit parameters, no hidden state;
//   - every kernel has a small-shape unit test in tests/.

namespace tinyqwen {

// y = x / sqrt(mean(x^2) + eps) * weight   (Qwen RMSNorm, weight-only)
void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps);

// y = W @ x, W row-major [out_dim, in_dim] (HF layout, NOT transposed).
void matvec_f32_ref(const float* w, const float* x, float* y, int out_dim, int in_dim);

// Numerically stable softmax: y[i] = exp(x[i] - max(x)) / sum_j exp(x[j] - max(x)).
void softmax_ref(const float* x, float* y, int n);

// y = x * sigmoid(x)
void silu_ref(const float* x, float* y, int n);

// Index of the first maximum.
int argmax_ref(const float* logits, int n);

// In-place rotary position embedding for one decode position.
// q: [n_heads * head_dim], k: [n_kv_heads * head_dim].
// Matches HF Qwen2 rotate-half convention:
//   out[i]        = x[i] * cos - x[i + half] * sin
//   out[i + half] = x[i + half] * cos + x[i] * sin
// with inv_freq[i] = theta ^ (-2i / head_dim), angle = pos * inv_freq[i].
void rope_ref(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta);

// Single-position (decode) attention over a cached KV.
//   q       : [n_heads * head_dim]
//   k_cache : [n_kv_heads][max_seq_len][head_dim], positions [0, seq_len) valid
//   v_cache : same layout
//   out     : [n_heads * head_dim]
// score = dot(q_h, k_{kv(h), t}) * scale; online softmax, no scratch memory.
// GQA mapping: kv head of query head h is h / (n_heads / n_kv_heads).
void attention_decode_ref(const float* q, const float* k_cache, const float* v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float* out);

}  // namespace tinyqwen
