#include "ref_ops.h"

namespace tinyqwen {

int argmax_ref(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

}  // namespace tinyqwen
