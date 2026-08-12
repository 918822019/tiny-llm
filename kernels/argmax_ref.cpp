// greedy 选择：最大 logit 的下标。
//
// 平局约定：取第一个最大值（严格 '>' 比较）。与 numpy/torch 的 argmax
// 一致，单元测试里有断言；将来用 SIMD 重写时必须保持同样行为，
// 否则 greedy 结果不可复现。

#include "ref_ops.h"

namespace tinyqwen {

int argmax_ref(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (logits[i] > logits[best]) best = i;  // 严格 '>'：平局取第一个
  }
  return best;
}

}  // namespace tinyqwen
