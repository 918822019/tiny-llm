// softmax：把一组任意实数变成一组"加起来等于 1 的概率"。
// 公式：softmax(x_i) = exp(x_i) / sum_j exp(x_j)。
// 分数越大的项，分到的概率越大。
//
// 数值稳定技巧：直接算 exp(大数) 会上溢（无穷大）。解决办法是先把每个数
// 减去最大值 m——这不改变结果（分子分母同时乘了 e^-m 抵消），但保证 exp 的
// 输入 <= 0，永远不会上溢。
//
// 注意：decode 时的 attention 不调用本函数，它在 attention_decode_ref 里
// 用 online softmax 融合了。本 kernel 是独立 reference，供单测和将来采样用。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void softmax_ref(const float* x, float* y, int n) {
  // 第 1 遍：找最大值，用于数值稳定。
  float m = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > m) {
      m = x[i];
    }
  }

  // 第 2 遍：算 exp(x - m) 并累加分母。用 double 累加减少舍入漂移
  //（Qwen2.5 词表有 151936 这么大，连加误差不可忽视）。
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i] - m);
    sum += y[i];
  }

  // 第 3 遍：除以分母归一化。用"乘倒数"只做一次除法，更快。
  const float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) {
    y[i] = y[i] * inv;
  }
}

}  // namespace tinyqwen
