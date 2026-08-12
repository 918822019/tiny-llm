// SiLU（swish）激活：y = x * sigmoid(x) = x / (1 + exp(-x))。
//
// 只用于 SwiGLU FFN 的 gate 分支（见 qwen_model.cpp：
// down_proj(silu(gate) * up)）。除法形式不会溢出：x 为大负数时
// exp(-x) -> +inf，结果趋于 0，行为正确。支持 in-place 调用（x == y）。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {

void silu_ref(const float* x, float* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
}

}  // namespace tinyqwen
