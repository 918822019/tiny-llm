// Numerically stable softmax: subtract max(x) before exp so large logits
// (e.g. untrained lm_head outputs) cannot overflow exp(). Subtracting a
// constant does not change the result mathematically.
//
// Note: the decode-time attention does NOT call this — attention_decode_ref
// fuses an online softmax over the KV positions instead. This kernel is the
// standalone reference used by unit tests and any future sampling path.

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void softmax_ref(const float* x, float* y, int n) {
  // Pass 1: row max for numerical stability.
  float m = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  // Pass 2: y = exp(x - max), summing in double to reduce rounding drift
  // on large vocabs (151936 for Qwen2.5).
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i] - m);
    sum += y[i];
  }
  // Pass 3: normalize (multiply by reciprocal: one division total).
  const float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) y[i] *= inv;
}

}  // namespace tinyqwen
