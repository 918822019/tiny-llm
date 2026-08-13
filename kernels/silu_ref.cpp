// SiLU（又叫 swish）激活函数：y = x * sigmoid(x) = x / (1 + exp(-x))。
//
// 激活函数给网络引入"非线性"，否则多层线性变换叠起来还是线性的。
// SiLU 是平滑版的 ReLU，负值不会被一刀切成 0，而是柔和地压小。
//
// 在 Qwen 里它只用于 SwiGLU FFN 的 gate 支路：down_proj(silu(gate) * up)。
// 用除法形式 x/(1+exp(-x)) 而非 x*sigmoid(x)，是为了对大负数安全：
// 此时 exp(-x) 趋于 +inf，结果自然趋于 0，不会出错。
// 支持 in-place（输入输出同一块内存）。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {
  void silu_ref(const float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) {
      const float denom = 1.0f + std::exp(-x[i]); // 分母 = 1 + exp(-x)
      y[i] = x[i] / denom; // y = x / (1 + exp(-x))
    }
  }
} // namespace tinyqwen
