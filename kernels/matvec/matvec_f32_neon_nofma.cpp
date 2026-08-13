// 矩阵乘向量：y = W @ x —— NEON 向量化版，**故意不用 FMA**。
//
// ============================================================================
// 归因阶梯上的位置（每层只加一个技术，A/B 才能说清功劳归谁）：
//
//   ref（标量、double 累加）
//     → double_2_float      [+ float 累加]
//     → acc4                [+ 4 条独立累加链]
//     → neon_nofma（本文件） [+ NEON 向量化]     ← 隔离"SIMD 宽度"
//     → neon                [+ FMA 融合]
//
// 本文件在 acc4 的基础上只加一件事：SIMD 向量化（一条指令处理 4 个
// float）。为了把"宽度"的贡献单独量出来，这里**刻意不用 FMA**——
// 乘和加拆成两条指令：
//
//     完整版 neon：acc = vfmaq_f32(acc, a, b)               // 乘加融合
//     本文件：     acc = vaddq_f32(acc, vmulq_f32(a, b))    // 乘、加分开
//
// 于是相邻两层 A/B 各 isolates 一个变量：
//     acc4 → neon_nofma 的差异 = 纯 SIMD 宽度（4 链结构完全相同）；
//     neon_nofma → neon 的差异 = 纯 FMA 融合（其余完全相同）。
//
// 除循环里那一条指令外，4 链结构、尾段处理、合并归约与 neon 文件
// 一一对应；逐行讲解（SIMD/FMA/累加 三概念拆解、命名规则、延迟
// 重叠原理）见 matvec_f32_neon.cpp 的注释。
//
// 数值说明：乘、加分开 = 两次舍入（FMA 只有一次），误差比 neon 略大，
// 仍与 ref 同量级，单测容差内对齐。
//
// 平台说明：与 neon 相同，仅 aarch64 编译注册；其他平台本文件为空，
// "neon_nofma" 不存在于注册表。
// 选用：--matvec-impl neon_nofma。不加 bias；与 ref 相同。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cstddef>

namespace tinyqwen {
  namespace {
    // W 的一行与 x 的点积：NEON 4 链，乘加**不融合**。
    inline float dot_row_neon_nofma(const float *row, const float *x, int n) {
      // 4 个独立累加器（结构与 acc4/neon 完全相同）。
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);

      int i = 0;
      // 主循环：一次迭代吃 16 个元素（4 链 × 4 lane），与 neon 相同。
      const int n16 = n & ~15;
      for (; i < n16; i += 16) {
        // 与 neon 唯一的区别就在这一类语句：
        //   vmulq_f32(a, b)：逐 lane 相乘，a[lane]*b[lane]（只乘不加）；
        //   vaddq_f32(acc, t)：逐 lane 相加，把乘积加进累加器。
        // 两条指令、两次舍入——这就是"不用 FMA"的全部含义。
        acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(row + i), vld1q_f32(x + i)));
        acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(row + i + 4), vld1q_f32(x + i + 4)));
        acc2 = vaddq_f32(acc2, vmulq_f32(vld1q_f32(row + i + 8), vld1q_f32(x + i + 8)));
        acc3 = vaddq_f32(acc3, vmulq_f32(vld1q_f32(row + i + 12), vld1q_f32(x + i + 12)));
      }
      // 向量尾段：还剩 4~15 个元素时按 4 个处理（同 neon）。
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(row + i), vld1q_f32(x + i)));
      }
      // 合并 + 横向归约：与 neon 完全相同（纯累加，不含乘法）。
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      // 标量尾段：不足 4 个的零头（同 neon）。
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }
  } // namespace

  void matvec_f32_neon_nofma(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; ++o) {
      const float *row = w + static_cast<size_t>(o) * in_dim;
      y[o] = dot_row_neon_nofma(row, x, in_dim);
    }
  }

  // 自注册进 dispatch：--matvec-impl neon_nofma 即可选用（仅 aarch64 构建存在）。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_nofma, "neon_nofma");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
