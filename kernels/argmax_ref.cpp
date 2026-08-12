// argmax：找出数组里最大元素的下标。
//
// 在推理里，logits 是"词表中每个词的分数"，argmax 就是"选分数最高的那个词"。
// 这种永远选最大值的策略叫 greedy（贪心）解码。
//
// 平局约定：如果有多个并列最大值，取第一个（用严格 '>' 保证）。
// 这与 numpy/torch 的 argmax 行为一致，且单测有断言；将来用 SIMD 重写时
// 必须保持同样行为，否则生成结果不可复现。

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
