#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void softmax_ref(const float* x, float* y, int n) {
  float m = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i] - m);
    sum += y[i];
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) y[i] *= inv;
}

}  // namespace tinyqwen
