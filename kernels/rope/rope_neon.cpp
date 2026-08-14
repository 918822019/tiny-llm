// RoPE 的 NEON 版（aarch64）。
//
// 与 rope_ref 的唯一区别在"施加旋转"的内循环：ref 逐对标量算
//   x0' = x0*c - x1*s；x1' = x1*c + x0*s
// 这里用 NEON 一次处理 4 个配对。cos/sin 表的计算（pow/cos/sin）与 ref
// 逐字相同（标量 libm），保证喂给旋转的数据一致；旋转本身用 vmulq + vsubq /
// vaddq（乘、加/减分开，不引入 FMA），尽量贴近 ref 的逐元素舍入。
//
// 数值差异：仅来自标量 ref 可能被编译器做 FP 收缩（fma）与本实现的乘加分离
// 之间的 ulp 级差别，单测按容差门禁（rel 1e-5）对齐 ref。rotate-half 约定、
// GQA 的 head 遍历、pos/theta 语义与 ref 完全一致。

#include "dispatch.h" // TINYQWEN_ROPE_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tinyqwen {
  namespace {
    void rope_neon(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
                   float theta) {
      const int half = head_dim / 2;
      constexpr int kMaxHalf = 256; // 与 rope_ref 一致
      if (half <= 0 || half > kMaxHalf) {
        std::fprintf(stderr, "rope_neon: unsupported head_dim %d\n", head_dim);
        std::abort();
      }

      // cos/sin 表：与 ref 逐字相同（所有 head 共享）。
      float cs[kMaxHalf];
      float sn[kMaxHalf];
      for (int i = 0; i < half; ++i) {
        const float exponent = -static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float inv_freq = std::pow(theta, exponent);
        const float angle = static_cast<float>(pos) * inv_freq;
        cs[i] = std::cos(angle);
        sn[i] = std::sin(angle);
      }

      // 对单个 head 施加旋转（NEON，就地）。
      const auto apply = [&](float *x) {
        int i = 0;
        for (; i + 4 <= half; i += 4) {
          const float32x4_t x0 = vld1q_f32(x + i);
          const float32x4_t x1 = vld1q_f32(x + i + half);
          const float32x4_t c = vld1q_f32(cs + i);
          const float32x4_t s = vld1q_f32(sn + i);
          // x0' = x0*c - x1*s；x1' = x1*c + x0*s（乘加分离，贴近 ref 舍入）。
          const float32x4_t r0 = vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s));
          const float32x4_t r1 = vaddq_f32(vmulq_f32(x1, c), vmulq_f32(x0, s));
          vst1q_f32(x + i, r0);
          vst1q_f32(x + i + half, r1);
        }
        for (; i < half; ++i) { // 标量尾段
          const float x0 = x[i];
          const float x1 = x[i + half];
          const float c = cs[i];
          const float s = sn[i];
          x[i] = x0 * c - x1 * s;
          x[i + half] = x1 * c + x0 * s;
        }
      };

      for (int h = 0; h < n_heads; ++h) {
        apply(q + static_cast<size_t>(h) * head_dim);
      }
      for (int h = 0; h < n_kv_heads; ++h) {
        apply(k + static_cast<size_t>(h) * head_dim);
      }
    }
  } // namespace

  // 自注册进 rope 注册表，名 "neon"。仅 aarch64 构建存在。
  TINYQWEN_ROPE_VARIANT(rope_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
