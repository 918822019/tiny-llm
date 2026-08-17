// ============================================================================
// L2 归一化（就地）的 NEON 优化实现
// ============================================================================
//
// 数学公式：x[i] = x[i] / sqrt(Σ x[i]² + eps)
//
// 实际维度（Qwen3.5-0.8B）：
//   n = 128（qk_head_dim），每层调用 32 次（16 q头 + 16 k头）。
//
// 优化策略：
//   ① vrsqrteq_f32 + 2 步 Newton-Raphson 替代标量 sqrt + div：
//      标量 sqrt(15cyc) + div(15cyc) = ~30 cycles；
//      NEON rsqrt 估计(3cyc) + 2×Newton(4cyc) = ~11 cycles，且无标量/向量域切换。
//      2 步 Newton 精度 ~24 bits，超过 fp32 的 23 bit 尾数，满足需求。
//   ② Pass 1（平方和）：4 路累加器隐藏 FMA 延迟，16 元素/迭代
//   ③ Pass 2（归一化）：16 元素/迭代展开（原 4 元素），减少循环开销
//      n=128 时 32 次循环 → 8 次，循环开销占比从 ~40% 降到 ~10%
//
// 数值精度：
//   - float 累加 vs ref 的 double：128 维下相对误差 ~1e-7
//   - vrsqrte + Newton 的额外误差 < 1 ULP
//   - 单测容差 1e-5
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>

namespace tinyqwen {
    namespace {
        void l2norm_inplace_neon(float *x, int n, float eps) {
            // ============================================================
            // Pass 1：平方和累加
            // 4 路独立累加器隐藏 FMA 的 3-4 cycle 延迟链
            // ============================================================
            float32x4_t s0 = vdupq_n_f32(0.0f);
            float32x4_t s1 = vdupq_n_f32(0.0f);
            float32x4_t s2 = vdupq_n_f32(0.0f);
            float32x4_t s3 = vdupq_n_f32(0.0f);

            int i = 0;
            for (; i + 16 <= n; i += 16) {
                float32x4_t a = vld1q_f32(x + i);
                float32x4_t b = vld1q_f32(x + i + 4);
                float32x4_t c = vld1q_f32(x + i + 8);
                float32x4_t d = vld1q_f32(x + i + 12);
                s0 = vfmaq_f32(s0, a, a);
                s1 = vfmaq_f32(s1, b, b);
                s2 = vfmaq_f32(s2, c, c);
                s3 = vfmaq_f32(s3, d, d);
            }
            for (; i + 4 <= n; i += 4) {
                float32x4_t a = vld1q_f32(x + i);
                s0 = vfmaq_f32(s0, a, a);
            }
            float sumsq = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            for (; i < n; ++i) sumsq += x[i] * x[i];

            // ============================================================
            // 计算 inv_norm = 1/sqrt(sumsq + eps)
            // 用 NEON vrsqrte + 2 步 Newton 替代标量 sqrt+div，
            // 避免标量/向量域切换带来的流水线停顿。
            // Newton 公式：est = est * (3 - val*est²) / 2
            // AArch64 提供 vrsqrtsq_f32(a,b) = (3-a*b)/2 直接用。
            // ============================================================
            const float val = sumsq + eps;
            float32x4_t val_v = vdupq_n_f32(val);
            float32x4_t est = vrsqrteq_f32(val_v);
            // Newton 第 1 步：精度 ~12 bits → ~24 bits
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val_v, est), est));
            // Newton 第 2 步：精度 ~24 bits（超过 fp32 尾数精度）
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val_v, est), est));
            const float32x4_t vs = vdupq_n_f32(vgetq_lane_f32(est, 0));

            // ============================================================
            // Pass 2：归一化 x[i] *= inv_norm
            // 16 元素/迭代展开，n=128 时仅 8 次循环
            // ============================================================
            int j = 0;
            for (; j + 16 <= n; j += 16) {
                vst1q_f32(x + j,      vmulq_f32(vld1q_f32(x + j),      vs));
                vst1q_f32(x + j + 4,  vmulq_f32(vld1q_f32(x + j + 4),  vs));
                vst1q_f32(x + j + 8,  vmulq_f32(vld1q_f32(x + j + 8),  vs));
                vst1q_f32(x + j + 12, vmulq_f32(vld1q_f32(x + j + 12), vs));
            }
            for (; j + 4 <= n; j += 4) {
                vst1q_f32(x + j, vmulq_f32(vld1q_f32(x + j), vs));
            }
            // 标量尾段
            const float inv_norm = vgetq_lane_f32(est, 0);
            for (; j < n; ++j) x[j] *= inv_norm;
        }
    } // namespace

    TINYQWEN_L2NORM_INPLACE_VARIANT(l2norm_inplace_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__
