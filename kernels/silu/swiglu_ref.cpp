// ============================================================================
// swiglu_ref.cpp — SwiGLU 融合算子的参考实现（标量版）
// ============================================================================
// 本文件实现 SwiGLU 融合算子：gate[i] = silu(gate[i]) * up[i]（就地修改 gate）。
//
// 数学定义：
//   SwiGLU(gate, up) = SiLU(gate) ⊙ up
//   其中 ⊙ 表示逐元素乘法，SiLU(x) = x / (1 + exp(-x))。
//   等价于两步操作：先 gate = silu(gate)，再 gate = gate * up。
//
// 在 Transformer 中的位置：
//   FFN（前馈网络）的核心计算，Qwen 使用 SwiGLU 作为 FFN 激活：
//     ffn_output = down_proj( SwiGLU(gate_proj(x), up_proj(x)) )
//   gate_proj 和 up_proj 的输出维度为 inter_size（Qwen2.5-0.5B 为 4864），
//   SwiGLU 融合避免了中间结果的额外内存读写。
//
// ref 与 neon 版本的关系：
//   - 本文件是标量参考实现，作为正确性锚点（golden reference）。
//   - swiglu_neon.cpp 用 NEON 向量化 + 多项式逼近 exp，单测按容差门禁对齐。
//   - 同时也是 ops dispatch 在未注册变体时的兜底（全平台可用）。
//
// 优化要点：
//   1. 将 silu + 逐元素乘合为一个 kernel，减少一次完整的内存遍历。
//   2. 使用与 silu_ref 完全相同的除法形式，保证逐位一致。
//   3. 就地修改 gate 数组，不需要额外的输出缓冲区。
// ============================================================================

#include "ref_ops.h" // 声明 swiglu_ref 等参考算子的头文件

#include <cmath> // std::exp

namespace tinyqwen {
  // =========================================================================
  // swiglu_ref — SwiGLU 融合算子参考实现
  // =========================================================================
  // 功能：对 gate 数组就地执行 SwiGLU：gate[i] = silu(gate[i]) * up[i]
  // 参数：
  //   gate — gate 投影的输出数组，长度 n；既是输入也是输出（就地修改）
  //   up   — up 投影的输出数组，长度 n（只读）
  //   n    — 数组长度（通常为 inter_size = 4864）
  void swiglu_ref(float *gate, const float *up, int n) {
    // 逐元素计算 SwiGLU
    for (int i = 0; i < n; ++i) {
      // 计算 SiLU 的分母 = 1 + exp(-gate[i])
      // 与 silu_ref 完全相同的除法形式（对大负数安全）
      const float denom = 1.0f + std::exp(-gate[i]);
      // 计算 silu(gate[i]) = gate[i] / (1 + exp(-gate[i]))
      const float s = gate[i] / denom;
      // SwiGLU = silu(gate[i]) * up[i]，结果写回 gate[i]（就地）
      gate[i] = s * up[i];
    }
  }
} // namespace tinyqwen
