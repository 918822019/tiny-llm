// RMSNorm (Qwen variant): y = x / sqrt(mean(x^2) + eps) * weight.
//
// - weight-only (no bias), matching Qwen2/2.5 layernorm modules;
// - sum of squares accumulates in double, then the result returns to fp32:
//   this stays closer to the PyTorch fp32 reference than naive fp32
//   accumulation over ~900+ elements;
// - used for input_layernorm, post_attention_layernorm and the final norm.

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps) {
  double sumsq = 0.0;  // double accumulation: closer to the fp32 ground truth
  for (int i = 0; i < n; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  // Single fused scale: 1 / rms. eps sits INSIDE the sqrt (HF definition).
  const float scale = 1.0f / std::sqrt(static_cast<float>(sumsq / n) + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

}  // namespace tinyqwen
