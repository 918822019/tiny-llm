// Rotary position embedding (reference, in-place).
//
// Pairing convention (must match HF Qwen2 "rotate_half"):
//   the rotation partner of x[i] is x[i + head_dim/2] — the head vector is
//   split into two halves, NOT interleaved pairs. Getting this wrong still
//   produces plausible output but silently diverges from PyTorch.
//
// cos/sin are computed once per position (shared by all heads), which matches
// HF's apply_rotary_pos_emb semantics.

#include "ref_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

void rope_ref(float* q, float* k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta) {
  const int half = head_dim / 2;
  constexpr int kMaxHalf = 256;  // head_dim <= 512, far above any Qwen config
  if (half <= 0 || half > kMaxHalf) {
    std::fprintf(stderr, "rope_ref: unsupported head_dim %d\n", head_dim);
    std::abort();
  }

  // inv_freq[i] = theta ^ (-2i / head_dim); angle = pos * inv_freq[i]
  float cs[kMaxHalf];
  float sn[kMaxHalf];
  for (int i = 0; i < half; ++i) {
    const float inv_freq =
        std::pow(theta, -static_cast<float>(2 * i) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * inv_freq;
    cs[i] = std::cos(angle);
    sn[i] = std::sin(angle);
  }

  // Apply the 2D rotation to every (x0, x1) pair of one head.
  const auto apply = [&](float* x) {
    for (int i = 0; i < half; ++i) {
      const float x0 = x[i];
      const float x1 = x[i + half];
      x[i] = x0 * cs[i] - x1 * sn[i];
      x[i + half] = x1 * cs[i] + x0 * sn[i];
    }
  };

  for (int h = 0; h < n_heads; ++h) apply(q + static_cast<size_t>(h) * head_dim);
  for (int h = 0; h < n_kv_heads; ++h) apply(k + static_cast<size_t>(h) * head_dim);
}

}  // namespace tinyqwen
