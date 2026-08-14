// argmax 的 NEON 版（aarch64）：返回第一个最大值的下标（平局取靠前者）。
//
// 为什么是 decode 非 matvec 里的头号目标：greedy 每 token 都要对 151936 维
// logits 扫一遍找最大。标量版（argmax_ref）是"逐个比较 + 维护当前最大"，
// 一条串行依赖链（每步等上一步的比较结果），且对 ~600KB 数据只做 ~2.5 GB/s
// 的有效吞吐——既没喂饱带宽，也没用 SIMD。profile 里它是非 matvec 最大单项。
//
// 两遍法（经典 SIMD argmax 套路）：
//   pass 1：只求"最大值是多少"——vmaxq_f32 对 4 lane 并行取 max，vmaxvq_f32
//           横向归约。max 运算没有下标依赖，纯归约，SIMD 满载。
//   pass 2：再扫一遍找"第一个等于最大值的下标"——vceqq_f32 并行比较出掩码。
// logits 刚被 lm_head 写出、热在 L2，第二遍几乎不花 DRAM 带宽。
//
// 数值语义与 argmax_ref **逐位一致**：max 在浮点下是精确运算（无舍入），
// "第一个 == max 的位置"正是 ref 用严格 `>` 保留首个平局的语义（含 ±0 相等）。
// 因此本变体无需容差门禁，单测直接断言下标相等。

#include "dispatch.h" // TINYQWEN_ARGMAX_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>

namespace tinyqwen {
  namespace {
    int argmax_neon(const float *logits, int n) {
      if (n <= 0) return 0;

      // ---- pass 1：NEON 归约求最大值 ----
      float32x4_t vmax = vdupq_n_f32(-INFINITY);
      int i = 0;
      for (; i + 4 <= n; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(logits + i));
      }
      float m = vmaxvq_f32(vmax); // 4 lane 横向取 max
      for (; i < n; ++i) {
        m = logits[i] > m ? logits[i] : m; // 标量尾段
      }

      // ---- pass 2：找第一个等于最大值的下标 ----
      const float32x4_t vm = vdupq_n_f32(m);
      alignas(16) uint32_t mask[4];
      int j = 0;
      for (; j + 4 <= n; j += 4) {
        const uint32x4_t eq = vceqq_f32(vld1q_f32(logits + j), vm);
        vst1q_u32(mask, eq);
        for (int l = 0; l < 4; ++l) {
          if (mask[l]) return j + l;
        }
      }
      for (; j < n; ++j) {
        if (logits[j] == m) return j;
      }
      return 0; // 不可达：最大值必然存在（NaN 场景除外，logits 不会出现）
    }
  } // namespace

  // 自注册进 argmax 注册表，名 "neon"。仅 aarch64 构建存在。
  TINYQWEN_ARGMAX_VARIANT(argmax_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
