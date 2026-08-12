// Greedy selection: index of the maximum logit.
//
// Tie-breaking convention: the FIRST maximum wins (strict '>' comparison).
// This matches numpy/torch argmax and is asserted in the unit tests, so a
// future SIMD rewrite must keep the same behavior to stay reproducible.

#include "ref_ops.h"

namespace tinyqwen {

int argmax_ref(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (logits[i] > logits[best]) best = i;  // strict '>' => first max wins
  }
  return best;
}

}  // namespace tinyqwen
