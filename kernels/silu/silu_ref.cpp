// ============================================================================
// silu_ref.cpp — SiLU (Swish) 激活函数的参考实现（标量版）
// ============================================================================
// 本文件实现 SiLU（Sigmoid Linear Unit，又称 Swish）激活函数。
//
// 数学定义：
//   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//   其中 sigmoid(x) = 1 / (1 + exp(-x)) 是标准 logistic 函数。
//
// 在 Transformer 中的位置：
//   SiLU 在 Qwen 中仅用于 SwiGLU FFN 的 gate 支路：
//     output = down_proj( silu(gate_proj(x)) * up_proj(x) )
//   即先对 gate 投影做 SiLU 激活，再与 up 投影逐元素相乘，最后 down 投影。
//
// ref 与 neon 版本的关系：
//   - 本文件是标量参考实现，使用 std::exp 保证精度。
//   - swiglu_neon.cpp 将 SiLU 融合进 SwiGLU kernel 并用多项式逼近 exp，
//     数值差异 ~1e-7，单测按容差门禁对齐。
//
// 优化要点：
//   1. 使用除法形式 x/(1+exp(-x)) 而非 x*sigmoid(x)，原因是对大负数安全：
//      当 x → -∞ 时，exp(-x) → +∞，分母趋于 +∞，结果自然趋于 0，不会出错。
//      若用 x*sigmoid(x)，sigmoid(-∞) 可能因浮点下溢变为 0，x*(-0) 得 NaN。
//   2. 支持 in-place（输入输出同一块内存），节省内存分配。
// ============================================================================

#include "ref_ops.h" // 声明 silu_ref 等参考算子的头文件

#include <cmath> // std::exp

namespace tinyqwen {
  // =========================================================================
  // silu_ref — SiLU 激活函数参考实现
  // =========================================================================
  // 功能：对长度为 n 的输入数组逐元素计算 SiLU(x) = x / (1 + exp(-x))
  // 参数：
  //   x — 输入数组，长度 n
  //   y — 输出数组，长度 n（可与 x 相同，即 in-place）
  //   n — 数组长度
  void silu_ref(const float *x, float *y, int n) {
    // 逐元素计算 SiLU
    for (int i = 0; i < n; ++i) {
      // 计算分母 = 1 + exp(-x[i])
      // 当 x[i] 为大正数时，exp(-x[i]) ≈ 0，分母 ≈ 1，结果 ≈ x[i]
      // 当 x[i] 为大负数时，exp(-x[i]) → +∞，分母 → +∞，结果 → 0
      const float denom = 1.0f + std::exp(-x[i]);
      // y[i] = x[i] / (1 + exp(-x[i]))，即 SiLU(x[i])
      y[i] = x[i] / denom;
    }
  }
} // namespace tinyqwen
