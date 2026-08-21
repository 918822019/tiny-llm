// ============================================================================
// rmsnorm_gated_neon.cpp — 门控 RMSNorm 的 NEON 优化实现
// ============================================================================
//
// 本文件实现 GDN（Gated DeltaNet）线性注意力层输出端的门控 RMSNorm 的
// NEON 加速版本。在 Qwen3.5 中，每个 GDN head 的输出经过此算子进行
// 归一化和动态门控调节。
//
// 数学公式：y[i] = (x[i] / sqrt(mean(x²) + eps) * weight[i]) * silu(gate[i])
// 其中 silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
//
// 分解为两步：
//   Step 1: normed[i] = x[i] * scale * weight[i]    （RMSNorm + 可学习缩放）
//   Step 2: y[i] = normed[i] * silu(gate[i])         （门控调制）
// 其中 scale = 1 / sqrt(mean(x²) + eps) = 1 / sqrt(sumsq/n + eps)
//
// 实际维度（Qwen3.5-0.8B）：
//   n = 128（v_head_dim），每层调用 16 次（16 个 GDN head）。
//
// 优化策略：
//   ① vrsqrteq_f32 + 2 步 Newton-Raphson 替代标量 sqrt+div 计算 scale
//      （与 l2norm_inplace_neon 相同策略，详见该文件注释）。
//   ② SiLU 保留 vdivq_f32：Apple Silicon 延迟仅 ~7 cycles，
//      vrecpe+Newton 在此平台反而更慢（Cortex-A78 上可考虑替换）。
//   ③ Pass 2 双路展开（8 元素/迭代）：两组独立的 exp+recip 管线，
//      乱序执行器可重叠一组的 exp 多项式延迟和另一组的 load/store。
//   ④ Pass 1 保持 4 路累加器 + 16 元素/迭代（已是最优结构）。
//
// exp 近似（vexpq_f32）：
//   整数取整分离指数部分 + 6 阶泰勒展开小数部分。
//   相对误差 < 2e-7，与 swiglu_neon / causal_conv1d_update_neon 共用算法。
//
// 数值精度：
//   - float 平方和 + exp 多项式 + recip Newton vs ref（double + std::exp）
//   - 综合误差 ~1e-5
//   - 单测容差 1e-4
// ============================================================================

#include "dispatch.h"  // 分发层头文件：提供 TINYQWEN_RMSNORM_GATED_VARIANT 宏

#if defined(__aarch64__) || defined(_M_ARM64)  // 仅在 AArch64 平台编译

#include <arm_neon.h>  // ARM NEON intrinsic 头文件
#include <cmath>       // std::exp（标量尾段使用）

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部函数不导出符号表
        // ================================================================
        // 向量化 exp：整数取整得 2^n 部分 + 6 阶泰勒展开小数部分
        // 与 swiglu_neon / causal_conv1d_update_neon 共用同一算法。
        // 精度：相对误差 < 2e-7（覆盖 [-87, 88] 有效范围）。
        //
        // 算法原理：将 exp(x) 分解为 2^n × exp(f*ln2)，其中
        //   n = round(x/ln2)，|f| ≤ 0.5
        // 2^n 通过 IEEE 754 位操作直接构造，exp(f*ln2) 用 Horner 法则求值。
        // ================================================================
        constexpr float kLog2e = 1.4426950408889634f;  // log2(e)，自然对数→以2为底的转换因子
        constexpr float kLn2 = 0.6931471805599453f;    // ln(2)，小数部分的缩放因子

        inline float32x4_t vexpq_f32(float32x4_t x) {
            // y = x * log2(e)：将输入从自然对数域转到以 2 为底
            const float32x4_t y = vmulq_f32(x, vdupq_n_f32(kLog2e));
            // magic number 取整技巧：2^23 * 1.5 = 12582912.0
            // 加到浮点数上再减去，利用 IEEE 754 舍入得到最近整数
            const float32x4_t magic = vdupq_n_f32(12582912.0f);
            const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);  // n = round(y)
            const float32x4_t f = vsubq_f32(y, n_f);                        // f = y - n（小数部分）

            // 通过 IEEE 754 位操作构造 2^n
            int32x4_t n_i = vcvtq_s32_f32(n_f);              // 浮点整数 → int32
            n_i = vmaxq_s32(n_i, vdupq_n_s32(-126));         // 钳制下界（最小规格化指数）
            n_i = vminq_s32(n_i, vdupq_n_s32(127));          // 钳制上界（最大规格化指数）
            // IEEE 754 single: 指数域 = n + 127（bias），左移 23 位到指数位置
            const float32x4_t pow2n = vreinterpretq_f32_s32(
                vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23));

            // 6 阶泰勒展开 exp(f*ln2) ≈ Σ (f*ln2)^k / k!，k=0..6
            // Horner 法则从高阶到低阶嵌套求值
            const float32x4_t t = vmulq_f32(f, vdupq_n_f32(kLn2));  // t = f * ln2
            float32x4_t p = vdupq_n_f32(1.0f / 720.0f);             // c6 = 1/6!
            p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), t, p);       // c5 + t*c6
            p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), t, p);        // c4 + t*(c5+t*c6)
            p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), t, p);         // c3 + t*(...)
            p = vfmaq_f32(vdupq_n_f32(0.5f), t, p);                 // c2 + t*(...)
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // c1 + t*(...)
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // c0 + t*(...)

            return vmulq_f32(pow2n, p);  // 2^n × poly(f*ln2)
        }

        // ================================================================
        // 向量化 SiLU：silu(g) = g / (1 + exp(-g))
        // Apple Silicon 的 vdivq_f32 延迟仅 ~7 cycles（远低于 Cortex-A78
        // 的 12-15 cycles），vrecpe+Newton 在此平台反而更慢，故保留精确除法。
        // ================================================================
        inline float32x4_t vsiluq_f32(float32x4_t g) {
            const float32x4_t one = vdupq_n_f32(1.0f);           // 常量 1.0
            float32x4_t e = vexpq_f32(vnegq_f32(g));              // exp(-g)
            return vdivq_f32(g, vaddq_f32(one, e));               // g / (1 + exp(-g))
        }

        // ================================================================
        // rmsnorm_gated_neon — NEON 优化的门控 RMSNorm
        // ================================================================
        // 参数：
        //   x      — 输入向量 [n]
        //   gate   — 门控向量 [n]（经 SiLU 激活后逐元素调制输出）
        //   weight — 可学习缩放权重 [n]
        //   y      — 输出向量 [n]
        //   n      — 向量维度（Qwen3.5 = 128）
        //   eps    — 防止除零的小常数
        void rmsnorm_gated_neon(const float *x, const float *gate, const float *weight,
                                float *y, int n, float eps) {
            // ============================================================
            // Pass 1：平方和累加（4 路独立累加器）
            // 与 l2norm_inplace_neon 相同的结构：4 路交错隐藏 FMA 延迟链。
            // ============================================================
            float32x4_t s0 = vdupq_n_f32(0.0f);  // 累加器 0
            float32x4_t s1 = vdupq_n_f32(0.0f);  // 累加器 1
            float32x4_t s2 = vdupq_n_f32(0.0f);  // 累加器 2
            float32x4_t s3 = vdupq_n_f32(0.0f);  // 累加器 3

            int i = 0;
            // 主循环：每次处理 16 个 float
            for (; i + 16 <= n; i += 16) {
                float32x4_t a = vld1q_f32(x + i);       // 加载 x[i:i+4]
                float32x4_t b = vld1q_f32(x + i + 4);   // 加载 x[i+4:i+8]
                float32x4_t c = vld1q_f32(x + i + 8);   // 加载 x[i+8:i+12]
                float32x4_t d = vld1q_f32(x + i + 12);  // 加载 x[i+12:i+16]
                s0 = vfmaq_f32(s0, a, a);  // s0 += a*a
                s1 = vfmaq_f32(s1, b, b);  // s1 += b*b
                s2 = vfmaq_f32(s2, c, c);  // s2 += c*c
                s3 = vfmaq_f32(s3, d, d);  // s3 += d*d
            }
            // 中间尾段：4 元素一组
            for (; i + 4 <= n; i += 4) {
                float32x4_t a = vld1q_f32(x + i);
                s0 = vfmaq_f32(s0, a, a);
            }
            // 水平归约：4 路累加器合并为标量
            float sumsq = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            // 标量尾段
            for (; i < n; ++i) sumsq += x[i] * x[i];

            // ============================================================
            // 计算 scale = 1/sqrt(mean(x²) + eps)
            // 使用 vrsqrte + 2 步 Newton-Raphson（同 l2norm 策略）
            // ============================================================
            const float rms_val = sumsq / static_cast<float>(n) + eps;  // mean(x²) + eps
            float32x4_t rms_v = vdupq_n_f32(rms_val);                   // 广播到 NEON
            float32x4_t est = vrsqrteq_f32(rms_v);                      // 硬件初始估计 ~12 bits
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(rms_v, est), est));  // Newton 第 1 步
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(rms_v, est), est));  // Newton 第 2 步
            // 提取并广播 scale 到 NEON 向量
            const float32x4_t vscale = vdupq_n_f32(vgetq_lane_f32(est, 0));

            // ============================================================
            // Pass 2：y = (x * scale * weight) * silu(gate)
            // 双路展开：两组独立 exp+recip 管线重叠执行。
            // 乱序执行器可在第 1 组执行 exp 多项式（~7 FMA 延迟链）时，
            // 发射第 2 组的 load/mul 指令，有效隐藏延迟。
            // ============================================================
            int j = 0;
            // 主循环：每次处理 8 个 float（2 组 × 4 元素）
            for (; j + 8 <= n; j += 8) {
                // 第 1 组：4 元素
                // xv0 = x[j:j+4] * scale * weight[j:j+4]（RMSNorm + 可学习权重）
                float32x4_t xv0 = vmulq_f32(vmulq_f32(vld1q_f32(x + j), vscale),
                                             vld1q_f32(weight + j));
                // silu0 = silu(gate[j:j+4])（门控激活）
                float32x4_t silu0 = vsiluq_f32(vld1q_f32(gate + j));

                // 第 2 组：4 元素（与第 1 组的 exp 延迟重叠执行）
                float32x4_t xv1 = vmulq_f32(vmulq_f32(vld1q_f32(x + j + 4), vscale),
                                             vld1q_f32(weight + j + 4));
                float32x4_t silu1 = vsiluq_f32(vld1q_f32(gate + j + 4));

                // 写出最终结果：y = xv * silu
                vst1q_f32(y + j,     vmulq_f32(xv0, silu0));   // y[j:j+4] = xv0 * silu0
                vst1q_f32(y + j + 4, vmulq_f32(xv1, silu1));   // y[j+4:j+8] = xv1 * silu1
            }
            // 单路 4 元素处理剩余部分
            for (; j + 4 <= n; j += 4) {
                float32x4_t xv = vmulq_f32(vmulq_f32(vld1q_f32(x + j), vscale),
                                            vld1q_f32(weight + j));
                float32x4_t silu_v = vsiluq_f32(vld1q_f32(gate + j));
                vst1q_f32(y + j, vmulq_f32(xv, silu_v));
            }
            // 标量尾段（n 非 4 整除时）
            const float scale = vgetq_lane_f32(est, 0);  // 提取标量 scale
            for (; j < n; ++j) {
                float normed = x[j] * scale * weight[j];                  // RMSNorm + 权重
                float s = gate[j] / (1.0f + std::exp(-gate[j]));         // 标量 SiLU
                y[j] = normed * s;                                        // 门控调制
            }
        }
    } // namespace

    // 自注册宏：将此 NEON 实现以 "neon" 名字登记到 rmsnorm_gated 分发注册表
    TINYQWEN_RMSNORM_GATED_VARIANT(rmsnorm_gated_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__ || _M_ARM64
