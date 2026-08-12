#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void silu_ref(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

}  // namespace tinyqwen
