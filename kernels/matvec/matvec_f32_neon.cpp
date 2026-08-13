// 矩阵乘向量：y = W @ x —— NEON SIMD 版（aarch64）。
//
// 数学含义与 matvec_f32_ref 完全一致：W 行主序 [out_dim, in_dim]，
// 每个输出分量是一行与 x 的连续内存点积。变的只是点积的算法：
//
//   1. 向量化：float32x4_t 一次装 4 个 float，vld1q_f32 加载
//      （NEON 加载不要求 16B 对齐，非对齐直接用）；
//   2. FMA：vfmaq_f32(acc, a, b) = acc + a*b，一条指令、只舍入一次
//      （aarch64 的 FMA 是基线指令，不需要额外编译选项）；
//   3. 4 路展开 + 4 个独立累加器：FMA 延迟约 3~4 周期，单条累加链
//      （acc = vfma(acc,...) 自依赖）会让流水线空转；4 条互不依赖的
//      链让乱序执行把延迟完全重叠。每次迭代 4×4 = 16 个元素在飞；
//   4. 主循环后合并 4 个累加器，vaddvq_f32 一条指令横向归约成标量；
//   5. 不足 4 个的尾段用标量补齐——不用 mask、不会越界读。
//
// 数值说明：float 累加，误差与 double_2_float 同阶；且 4 条链各自
// 只累加 in_dim/16 项，舍入链比逐元素串行更短。单测以 ref 为门禁。
//
// 平台说明：NEON 是 aarch64（Apple Silicon、Android arm64-v8a）的
// 基线指令集，无需 -march/-mfpu 选项。整个实现包在 __aarch64__ 里：
//   - aarch64 平台（本机、arm64 Android）：正常编译并自注册；
//   - 其他平台（如 x86）：本文件编译为空翻译单元，"neon" 不注册，
//     dispatch 自动兜底到 ref，构建与运行都不受影响。
//
// 选用：--matvec-impl neon / tinyqwen.conf 里 matvec_impl = neon。
// 注意：这里不加 bias；与 ref 相同。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cstddef>

namespace tinyqwen {
  namespace {
    // W 的一行与 x 的点积：整个 kernel 的内循环。
    inline float dot_row_neon(const float *row, const float *x, int n) {
      // 4 个独立累加器（见文件头注释第 3 点）。
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);

      int i = 0;
      // 主循环：一次迭代 4×4 = 16 个元素。
      // n16 = n 向下取整到 16 的倍数，保证 4 条链都不越界。
      const int n16 = n & ~15;
      for (; i < n16; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(row + i + 8), vld1q_f32(x + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
      }
      // 向量尾段：还剩 ≥ 4 个元素时按 4 个处理。
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
      }
      // 合并 4 条累加链，再横向归约成标量（vaddvq_f32 一条指令）。
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      // 标量尾段：不足 4 个的剩余元素，顺序累加即可（无越界读）。
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }
  } // namespace

  void matvec_f32_neon(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    // 外层循环：一行一行处理，与 ref 结构一致。
    for (int o = 0; o < out_dim; ++o) {
      const float *row = w + static_cast<size_t>(o) * in_dim;
      y[o] = dot_row_neon(row, x, in_dim);
    }
  }

  // 自注册进 dispatch：--matvec-impl neon 即可选用（仅 aarch64 构建存在）。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
