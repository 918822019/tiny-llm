// ============================================================================
// 门控 RMSNorm 的 NEON 优化实现
// ============================================================================
//
// 数学公式：y[i] = (x[i] / sqrt(mean(x²) + eps) * weight[i]) * silu(gate[i])
// 其中 silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
//
// 实际维度（Qwen3.5-0.8B）：
//   n = 128（v_head_dim），每层调用 16 次（16 个 GDN head）。
//
// 优化策略：
//   ① vrsqrteq_f32 + Newton 替代标量 sqrt+div 计算 scale（同 l2norm）
//   ② SiLU 保留 vdivq_f32（Apple Silicon 延迟仅 ~7 cycles，
//      vrecpe+Newton 在此平台反而更慢；Cortex-A78 可考虑 vrecpe 替代）
//   ③ Pass 2 双路展开（8 元素/迭代）：两组独立的 exp+recip 管线，
//      乱序执行器可重叠一组的 exp 多项式延迟和另一组的 load/store
//   ④ Pass 1 保持 4 路累加器 + 16 元素/迭代（已是最优结构）
//
// exp 近似（vexpq_f32）：
//   整数取整分离指数部分 + 6 阶泰勒展开小数部分。
//   相对误差 ~1e-7，与 swiglu_neon 共用同一实现。
//
// 数值精度：
//   - float 平方和 + exp 多项式 + recip Newton vs ref（double + std::exp）
//   - 综合误差 ~1e-5
//   - 单测容差 1e-4
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>

namespace tinyqwen {
    namespace {
        // ================================================================
        // 向量化 exp：整数取整得 2^n 部分 + 6 阶泰勒展开小数部分
        // 与 swiglu_neon / causal_conv1d_update_neon 共用同一算法。
        // 精度：相对误差 < 2e-7（覆盖 [-87, 88] 有效范围）。
        // ================================================================
        constexpr float kLog2e = 1.4426950408889634f;
        constexpr float kLn2 = 0.6931471805599453f;

        inline float32x4_t vexpq_f32(float32x4_t x) {
            // x = n*ln2 + f*ln2, 其中 n = round(x/ln2), |f| <= 0.5
            const float32x4_t y = vmulq_f32(x, vdupq_n_f32(kLog2e));
            // magic number 取整：加 2^23*1.5 再减，得到最近整数
            const float32x4_t magic = vdupq_n_f32(12582912.0f);
            const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);
            const float32x4_t f = vsubq_f32(y, n_f);

            // 2^n 通过位操作构造（IEEE 754 指数域）
            int32x4_t n_i = vcvtq_s32_f32(n_f);
            n_i = vmaxq_s32(n_i, vdupq_n_s32(-126));
            n_i = vminq_s32(n_i, vdupq_n_s32(127));
            const float32x4_t pow2n = vreinterpretq_f32_s32(
                vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23));

            // 6 阶泰勒：e^(f*ln2) ≈ 1 + t + t²/2 + t³/6 + t⁴/24 + t⁵/120 + t⁶/720
            const float32x4_t t = vmulq_f32(f, vdupq_n_f32(kLn2));
            float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
            p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), t, p);
            p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), t, p);
            p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), t, p);
            p = vfmaq_f32(vdupq_n_f32(0.5f), t, p);
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);

            return vmulq_f32(pow2n, p);
        }

        // ================================================================
        // 向量化 SiLU：g / (1 + exp(-g))
        // Apple Silicon 的 vdivq_f32 延迟仅 ~7 cycles（远低于 Cortex-A78
        // 的 12-15 cycles），vrecpe+Newton 在此平台反而更慢，故保留精确除法。
        // ================================================================
        inline float32x4_t vsiluq_f32(float32x4_t g) {
            const float32x4_t one = vdupq_n_f32(1.0f);
            float32x4_t e = vexpq_f32(vnegq_f32(g));
            return vdivq_f32(g, vaddq_f32(one, e));
        }

        void rmsnorm_gated_neon(const float *x, const float *gate, const float *weight,
                                float *y, int n, float eps) {
            // ============================================================
            // Pass 1：平方和累加（4 路独立累加器）
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

            // scale = 1/sqrt(mean(x²) + eps)，用 vrsqrte + Newton 计算
            const float rms_val = sumsq / static_cast<float>(n) + eps;
            float32x4_t rms_v = vdupq_n_f32(rms_val);
            float32x4_t est = vrsqrteq_f32(rms_v);
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(rms_v, est), est));
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(rms_v, est), est));
            const float32x4_t vscale = vdupq_n_f32(vgetq_lane_f32(est, 0));

            // ============================================================
            // Pass 2：y = (x * scale * weight) * silu(gate)
            // 双路展开：两组独立 exp+recip 管线重叠执行
            // ============================================================
            int j = 0;
            for (; j + 8 <= n; j += 8) {
                // 第 1 组：4 元素
                float32x4_t xv0 = vmulq_f32(vmulq_f32(vld1q_f32(x + j), vscale),
                                             vld1q_f32(weight + j));
                float32x4_t silu0 = vsiluq_f32(vld1q_f32(gate + j));

                // 第 2 组：4 元素（与第 1 组的 exp 延迟重叠执行）
                float32x4_t xv1 = vmulq_f32(vmulq_f32(vld1q_f32(x + j + 4), vscale),
                                             vld1q_f32(weight + j + 4));
                float32x4_t silu1 = vsiluq_f32(vld1q_f32(gate + j + 4));

                vst1q_f32(y + j,     vmulq_f32(xv0, silu0));
                vst1q_f32(y + j + 4, vmulq_f32(xv1, silu1));
            }
            for (; j + 4 <= n; j += 4) {
                float32x4_t xv = vmulq_f32(vmulq_f32(vld1q_f32(x + j), vscale),
                                            vld1q_f32(weight + j));
                float32x4_t silu_v = vsiluq_f32(vld1q_f32(gate + j));
                vst1q_f32(y + j, vmulq_f32(xv, silu_v));
            }
            // 标量尾段
            const float scale = vgetq_lane_f32(est, 0);
            for (; j < n; ++j) {
                float normed = x[j] * scale * weight[j];
                float s = gate[j] / (1.0f + std::exp(-gate[j]));
                y[j] = normed * s;
            }
        }
    } // namespace

    TINYQWEN_RMSNORM_GATED_VARIANT(rmsnorm_gated_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__
