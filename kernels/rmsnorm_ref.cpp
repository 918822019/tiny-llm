#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps) {
  double sumsq = 0.0;  // double accumulation: closer to the fp32 ground truth
  for (int i = 0; i < n; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  const float scale = 1.0f / std::sqrt(static_cast<float>(sumsq / n) + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

}  // namespace tinyqwen
