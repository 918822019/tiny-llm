// ============================================================================
// Causal depthwise conv1d 单步更新的 NEON 优化实现
// ============================================================================
//
// 算法（每通道独立）：
//   out[c] = silu( Σ_i weight[c][i] * [state_c..., x[c]][i] )
//   然后状态左移一格，推入 x[c]
//
// 实际维度（Qwen3.5-0.8B）：
//   dim = 6144, kernel_size = 4（state_len = 3），每层调用 1 次。
//
// 内存布局：
//   conv_state[dim][state_len=3]：通道主序，每通道 3 个历史输入（旧→新）
//   weight[dim][kernel_size=4]：通道主序，每通道 4 个卷积权重
//
// 优化策略：
//   ① 跨通道向量化：每 4 个通道一组 NEON 并行（通道间完全独立）
//   ② vld3q_f32 / vst3q_f32 解交错加载/存储 state（硬件 deinterleave）
//   ③ vld4q_f32 解交错加载 weight（4 元素 stride 天然对应 kernel_size=4）
//   ④ 双路 8 通道展开：两组独立的 dot+SiLU 管线重叠执行，
//      隐藏 exp 多项式的 7-FMA 延迟链
//   ⑤ SiLU 保留 vdivq（Apple Silicon 延迟低，vrecpe 无优势）
//
// 就地安全：x 和 out 可同址——先保存 xc 再写 out。
//
// 数值精度：
//   - exp 多项式 + recip Newton vs std::exp + 精确除法
//   - 相对误差 ~2e-7
//   - 单测容差 1e-5
// ============================================================================

#include "dispatch.h"
#include "gdn_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>

namespace tinyqwen {
    namespace {
        // ================================================================
        // 向量化 exp（6 阶泰勒，与 rmsnorm_gated_neon 共用算法）
        // ================================================================
        constexpr float kLog2e = 1.4426950408889634f;
        constexpr float kLn2 = 0.6931471805599453f;

        inline float32x4_t vexpq_f32(float32x4_t x) {
            const float32x4_t y = vmulq_f32(x, vdupq_n_f32(kLog2e));
            const float32x4_t magic = vdupq_n_f32(12582912.0f);
            const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);
            const float32x4_t f = vsubq_f32(y, n_f);

            int32x4_t n_i = vcvtq_s32_f32(n_f);
            n_i = vmaxq_s32(n_i, vdupq_n_s32(-126));
            n_i = vminq_s32(n_i, vdupq_n_s32(127));
            const float32x4_t pow2n = vreinterpretq_f32_s32(
                vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23));

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
        // Apple Silicon 的 vdivq_f32 延迟仅 ~7 cycles，
        // vrecpe+Newton 在此平台无优势（Cortex-A78 可考虑替换）。
        // ================================================================
        inline float32x4_t vsiluq_f32(float32x4_t g) {
            const float32x4_t one = vdupq_n_f32(1.0f);
            float32x4_t e = vexpq_f32(vnegq_f32(g));
            return vdivq_f32(g, vaddq_f32(one, e));
        }

        void causal_conv1d_update_neon(const float *x, float *conv_state,
                                       const float *weight, float *out,
                                       int dim, int kernel_size) {
            // 非 kernel_size=4 兜底到 ref（Qwen3.5 固定为 4，不应命中此分支）
            if (kernel_size != 4) {
                causal_conv1d_update_ref(x, conv_state, weight, out, dim, kernel_size);
                return;
            }

            const int state_len = 3; // kernel_size - 1

            // ============================================================
            // 主循环：双路 8 通道展开
            // 每次处理 8 个通道（2 组 × 4 通道），两组的 exp 延迟互相重叠。
            // ============================================================
            int c = 0;
            for (; c + 8 <= dim; c += 8) {
                // --- 第 1 组：通道 c ~ c+3 ---
                float32x4_t xc0 = vld1q_f32(x + c);
                float *sp0 = conv_state + c * state_len;
                float32x4x3_t st0 = vld3q_f32(sp0);
                float32x4x4_t wt0 = vld4q_f32(weight + c * kernel_size);

                float32x4_t acc0 = vmulq_f32(wt0.val[0], st0.val[0]);
                acc0 = vfmaq_f32(acc0, wt0.val[1], st0.val[1]);
                acc0 = vfmaq_f32(acc0, wt0.val[2], st0.val[2]);
                acc0 = vfmaq_f32(acc0, wt0.val[3], xc0);

                // --- 第 2 组：通道 c+4 ~ c+7（与第 1 组的 SiLU 延迟重叠）---
                float32x4_t xc1 = vld1q_f32(x + c + 4);
                float *sp1 = conv_state + (c + 4) * state_len;
                float32x4x3_t st1 = vld3q_f32(sp1);
                float32x4x4_t wt1 = vld4q_f32(weight + (c + 4) * kernel_size);

                float32x4_t acc1 = vmulq_f32(wt1.val[0], st1.val[0]);
                acc1 = vfmaq_f32(acc1, wt1.val[1], st1.val[1]);
                acc1 = vfmaq_f32(acc1, wt1.val[2], st1.val[2]);
                acc1 = vfmaq_f32(acc1, wt1.val[3], xc1);

                // SiLU（两组独立发射，OoO 可重叠 exp 多项式延迟）
                vst1q_f32(out + c,     vsiluq_f32(acc0));
                vst1q_f32(out + c + 4, vsiluq_f32(acc1));

                // 状态左移：st[0]=st[1], st[1]=st[2], st[2]=原始输入
                st0.val[0] = st0.val[1]; st0.val[1] = st0.val[2]; st0.val[2] = xc0;
                vst3q_f32(sp0, st0);

                st1.val[0] = st1.val[1]; st1.val[1] = st1.val[2]; st1.val[2] = xc1;
                vst3q_f32(sp1, st1);
            }

            // 单路 4 通道处理剩余部分
            for (; c + 4 <= dim; c += 4) {
                float32x4_t xc = vld1q_f32(x + c);
                float *sp = conv_state + c * state_len;
                float32x4x3_t st = vld3q_f32(sp);
                float32x4x4_t wt = vld4q_f32(weight + c * kernel_size);

                float32x4_t acc = vmulq_f32(wt.val[0], st.val[0]);
                acc = vfmaq_f32(acc, wt.val[1], st.val[1]);
                acc = vfmaq_f32(acc, wt.val[2], st.val[2]);
                acc = vfmaq_f32(acc, wt.val[3], xc);

                vst1q_f32(out + c, vsiluq_f32(acc));

                st.val[0] = st.val[1]; st.val[1] = st.val[2]; st.val[2] = xc;
                vst3q_f32(sp, st);
            }

            // 标量尾段（dim 非 4 整除时）
            for (; c < dim; ++c) {
                const float xc = x[c];
                float *st = conv_state + static_cast<size_t>(c) * state_len;
                const float *w = weight + static_cast<size_t>(c) * kernel_size;

                float acc = 0.0f;
                for (int i = 0; i < state_len; ++i) acc += w[i] * st[i];
                acc += w[state_len] * xc;

                float e = std::exp(-acc);
                out[c] = acc / (1.0f + e);

                for (int i = 0; i < state_len - 1; ++i) st[i] = st[i + 1];
                st[state_len - 1] = xc;
            }
        }
    } // namespace

    TINYQWEN_CAUSAL_CONV1D_UPDATE_VARIANT(causal_conv1d_update_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__
