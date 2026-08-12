// Decode-step attention (one query position against the cached KV).
//
// Uses the online-softmax recurrence (same idea as FlashDecoding), so the
// kernel needs O(1) scratch memory: no per-position score buffer. This keeps
// the reference allocation-free and already mirrors the access pattern a
// fused kernel will use later.

#include "ref_ops.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace tinyqwen {

void attention_decode_ref(const float* q, const float* k_cache, const float* v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float* out) {
  const int heads_per_kv = n_heads / n_kv_heads;  // GQA fan-out factor
  // Stride between two kv heads inside one layer block:
  // layout is [n_kv_heads][max_seq_len][head_dim].
  const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

  for (int h = 0; h < n_heads; ++h) {
    const int kv = h / heads_per_kv;
    const float* qh = q + static_cast<size_t>(h) * head_dim;
    const float* kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
    const float* vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
    float* oh = out + static_cast<size_t>(h) * head_dim;

    // Online softmax state. Invariant after processing positions [0, t):
    //   m  = max score seen so far
    //   l  = sum_j exp(s_j - m)
    //   oh = sum_j exp(s_j - m) * v_j        (unnormalized output)
    // A new maximum m_new rescales the old accumulation by exp(m - m_new),
    // which is exactly 1 until the first real score arrives (m == -inf).
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.0f;
    for (int i = 0; i < head_dim; ++i) oh[i] = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
      const float* kt = kh + static_cast<size_t>(t) * head_dim;
      const float* vt = vh + static_cast<size_t>(t) * head_dim;

      double dot = 0.0;
      for (int i = 0; i < head_dim; ++i) dot += static_cast<double>(qh[i]) * kt[i];
      const float s = static_cast<float>(dot) * scale;

      const float m_new = s > m ? s : m;
      const float rescale = std::exp(m - m_new);  // 1.0f when m == -inf
      const float p = std::exp(s - m_new);
      for (int i = 0; i < head_dim; ++i) oh[i] = oh[i] * rescale + p * vt[i];
      l = l * rescale + p;
      m = m_new;
    }

    const float inv_l = 1.0f / l;  // seq_len >= 1 always holds during decode
    for (int i = 0; i < head_dim; ++i) oh[i] *= inv_l;
  }
}

}  // namespace tinyqwen
