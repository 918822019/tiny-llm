// SwiGLU 融合的参考实现：gate[i] = silu(gate[i]) * up[i]（就地）。
//
// 它等价于 forward 里原来的两步：
//     silu_ref(gate, gate, inter);          // gate = silu(gate)
//     for j: gate[j] *= up[j];              // gate = gate * up
// 逐位一致——本文件就是把这两步合成一个 kernel，作为融合 op 的正确性锚点，
// 也是 ops dispatch 在未注册变体时的兜底（正确性基准 + 全平台可用）。
// NEON 优化版（swiglu_neon.cpp）用多项式逼近 sigmoid，需对本实现做容差门禁。

#include "ref_ops.h"

#include <cmath>

namespace tinyqwen {
  void swiglu_ref(float *gate, const float *up, int n) {
    for (int i = 0; i < n; ++i) {
      // 与 silu_ref 完全相同的除法形式（对大负数安全），先算 silu 再乘 up，
      // 保证与"silu_ref + 逐元素乘"两步调用逐位一致。
      const float denom = 1.0f + std::exp(-gate[i]);
      const float s = gate[i] / denom; // silu(gate[i])
      gate[i] = s * up[i];
    }
  }
} // namespace tinyqwen
