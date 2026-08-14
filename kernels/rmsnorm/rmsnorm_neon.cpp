// RMSNorm 的 NEON 版（aarch64）：y = x / sqrt(mean(x^2) + eps) * weight。
//
// ref 用 double 累加平方和（正确性锚点）。这里改用 NEON float 累加：
//   - pass 1：4 个独立 float32x4 累加器，一条指令平方 + 累加 4 lane，
//     循环展开到每轮 16 元素，隐藏 FMA 延迟；尾段标量补齐。
//   - pass 2：scale 归一化 + 乘 weight 两步都用 vmulq（与 ref 的逐元素
//     结合序一致：先 x*scale 再 *weight），纯向量化。
//
// 数值差异来源：pass 1 的 float 累加 vs ref 的 double 累加——平方和的舍入
// 差约在 ulp 级，传导到 scale 后对 y 的相对扰动 ~1e-7。远小于贪心决策的
// logit 间距，golden token 不受影响；单测按容差门禁（rel 1e-5）对齐 ref。

#include "dispatch.h" // TINYQWEN_RMSNORM_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>

namespace tinyqwen {
  namespace {
    void rmsnorm_neon(const float *x, const float *weight, float *y, int n, float eps) {
      // ---- pass 1：平方和（4 路独立累加器）----
      float32x4_t s0 = vdupq_n_f32(0.0f);
      float32x4_t s1 = vdupq_n_f32(0.0f);
      float32x4_t s2 = vdupq_n_f32(0.0f);
      float32x4_t s3 = vdupq_n_f32(0.0f);
      int i = 0;
      for (; i + 16 <= n; i += 16) {
        const float32x4_t a = vld1q_f32(x + i);
        const float32x4_t b = vld1q_f32(x + i + 4);
        const float32x4_t c = vld1q_f32(x + i + 8);
        const float32x4_t d = vld1q_f32(x + i + 12);
        s0 = vfmaq_f32(s0, a, a);
        s1 = vfmaq_f32(s1, b, b);
        s2 = vfmaq_f32(s2, c, c);
        s3 = vfmaq_f32(s3, d, d);
      }
      // 合并 4 路 + 处理不足 16 的剩余（先按 4，再标量）。
      float32x4_t ssum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
      for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        ssum = vfmaq_f32(ssum, v, v);
      }
      float sumsq = vaddvq_f32(ssum);
      for (; i < n; ++i) {
        sumsq += x[i] * x[i];
      }

      // ---- 与 ref 相同的标量公式：mean -> +eps -> sqrt -> 1/rms ----
      const float mean_sq = sumsq / static_cast<float>(n);
      const float rms = std::sqrt(mean_sq + eps);
      const float scale = 1.0f / rms;

      // ---- pass 2：y = (x*scale) * weight，向量化，结合序同 ref ----
      const float32x4_t vs = vdupq_n_f32(scale);
      int j = 0;
      for (; j + 4 <= n; j += 4) {
        const float32x4_t xv = vld1q_f32(x + j);
        const float32x4_t wv = vld1q_f32(weight + j);
        const float32x4_t normed = vmulq_f32(xv, vs);   // x * scale
        vst1q_f32(y + j, vmulq_f32(normed, wv));        // normed * weight
      }
      for (; j < n; ++j) {
        y[j] = (x[j] * scale) * weight[j];
      }
    }
  } // namespace

  // 自注册进 rmsnorm 注册表，名 "neon"。仅 aarch64 构建存在。
  TINYQWEN_RMSNORM_VARIANT(rmsnorm_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
