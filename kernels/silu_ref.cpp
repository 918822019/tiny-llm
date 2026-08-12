// SiLU (swish) activation: y = x * sigmoid(x) = x / (1 + exp(-x)).
//
// Used on the GATE branch of the SwiGLU FFN only (see qwen_model.cpp:
// down_proj(silu(gate) * up)). The division form is overflow-safe: for very
// negative x, exp(-x) -> +inf and the result tends to 0, which is correct.
// In-place calls (x == y) are supported.

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void silu_ref(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

}  // namespace tinyqwen
