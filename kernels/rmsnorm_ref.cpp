// RMSNorm（Qwen 变体）：y = x / sqrt(mean(x^2) + eps) * weight
//
// 它是 LayerNorm 的简化版，作用是"归一化"：把向量的幅度拉到一个稳定范围，
// 让后面的层更好训练/推理。做法是除以自身的均方根（RMS）。
//   - mean(x^2)：各分量平方后取平均；
//   - + eps：防止向量全零时除零（eps 是个极小正数）；
//   - * weight：每个分量再乘一个可学习的缩放系数（只有 weight，没有 bias）。
//
// 模型里 input_layernorm、post_attention_layernorm 和最后的 norm 都用它。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps) {
  // 第 1 遍：求平方和。用 double 累加减少误差（见 matvec 里的说明）。
  double sumsq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double xi = static_cast<double>(x[i]);
    sumsq += xi * xi;
  }

  // 平方和取平均，得 mean(x^2)。
  const float mean_sq = static_cast<float>(sumsq / n);

  // 加 eps 后开方得 RMS。注意 eps 在 sqrt 内部（这是 HF 的定义）。
  const float rms = std::sqrt(mean_sq + eps);

  // 融合成缩放系数 scale = 1 / RMS。
  const float scale = 1.0f / rms;

  // 第 2 遍：每个分量先归一化，再乘对应的 weight。
  for (int i = 0; i < n; ++i) {
    const float normed = x[i] * scale;
    y[i] = normed * weight[i];
  }
}

}  // namespace tinyqwen
