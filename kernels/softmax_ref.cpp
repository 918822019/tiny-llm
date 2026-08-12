// 数值稳定 softmax：exp 之前先减去 max(x)，避免大 logits
// （例如未训练 lm_head 的输出）导致 exp 上溢。减去常数在数学上不改变结果。
//
// 注意：decode 时的 attention 并不调用本函数——attention_decode_ref 内部
// 对 KV 位置做融合的 online softmax。本 kernel 是独立 reference，
// 供单元测试以及后续采样路径使用。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void softmax_ref(const float* x, float* y, int n) {
  // 第 1 遍：求行内最大值，保证数值稳定。
  float m = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  // 第 2 遍：y = exp(x - max)；大词表（Qwen2.5 为 151936）下
  // 用 double 求和减少舍入漂移。
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i] - m);
    sum += y[i];
  }
  // 第 3 遍：归一化（乘倒数，全程只做一次除法）。
  const float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) y[i] *= inv;
}

}  // namespace tinyqwen
