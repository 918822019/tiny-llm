// RMSNorm（Qwen 变体）：y = x / sqrt(mean(x^2) + eps) * weight。
//
// - 只有 weight、没有 bias，与 Qwen2/2.5 的 layernorm 模块一致；
// - 平方和用 double 累加，结果再回到 fp32：相比朴素 fp32 累加，
//   在 ~900+ 元素规模上更贴近 PyTorch fp32 参考实现；
// - input_layernorm、post_attention_layernorm 和 final norm 都用它。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void rmsnorm_ref(const float* x, const float* weight, float* y, int n, float eps) {
  double sumsq = 0.0;  // double 累加：更贴近 fp32 参考值
  for (int i = 0; i < n; ++i) sumsq += static_cast<double>(x[i]) * x[i];
  // 融合成一个 scale = 1 / rms；注意 eps 在 sqrt 内部（HF 定义）。
  const float scale = 1.0f / std::sqrt(static_cast<float>(sumsq / n) + eps);
  for (int i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

}  // namespace tinyqwen
