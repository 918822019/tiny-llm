// ============================================================================
// l2norm_inplace_neon.cpp — L2 归一化（就地）的 NEON 优化实现
// ============================================================================
//
// 本文件实现 GDN 线性注意力层中 q/k 向量的 L2 归一化的 NEON 加速版本。
// 在 Qwen3.5 中，每个 GDN head 的 query 和 key 向量在参与 delta rule
// 计算前需要 L2 归一化，以确保状态更新的数值稳定性。
//
// 数学公式：x[i] = x[i] / sqrt(Σ x[j]² + eps)
// 即：将向量除以其 L2 范数（加 eps 防除零），使结果为单位向量。
//
// 实际维度（Qwen3.5-0.8B）：
//   n = 128（qk_head_dim），每层调用 32 次（16 个 q 头 + 16 个 k 头）。
//   虽然单次计算量小，但调用频次高，仍需优化。
//
// 优化策略：
//   ① vrsqrteq_f32 + 2 步 Newton-Raphson 替代标量 sqrt + div：
//      标量路径：sqrt(~15 cyc) + div(~15 cyc) ≈ 30 cycles + 标量/向量域切换开销。
//      NEON 路径：vrsqrte 估计(~3 cyc) + 2×Newton(各~4 cyc) ≈ 11 cycles，
//      且全程在 NEON 域内完成，无域切换流水线停顿。
//      2 步 Newton 精度 ~24 bits，超过 fp32 的 23 bit 尾数，完全满足需求。
//   ② Pass 1（平方和累加）：4 路独立累加器隐藏 FMA 的 3-4 cycle 延迟链，
//      每次迭代处理 16 个 float。
//   ③ Pass 2（归一化乘法）：16 元素/迭代展开（vs 原始 4 元素），
//      n=128 时从 32 次循环降到 8 次，循环开销占比从 ~40% 降到 ~10%。
//
// 数值精度：
//   - float 累加 vs ref 的 double：128 维下相对误差 ~1e-7
//   - vrsqrte + 2 步 Newton 的额外误差 < 1 ULP
//   - 单测容差 1e-5
// ============================================================================

#include "dispatch.h"  // 分发层头文件：提供 TINYQWEN_L2NORM_INPLACE_VARIANT 宏

#if defined(__aarch64__) || defined(_M_ARM64)  // 仅在 AArch64 平台编译

#include <arm_neon.h>  // ARM NEON intrinsic 头文件
#include <cmath>       // 未直接使用，但保留以备将来 fallback

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部函数不导出符号表
        // ================================================================
        // l2norm_inplace_neon — NEON 优化的 L2 归一化（就地）
        // ================================================================
        // 参数：
        //   x   — 输入/输出向量 [n]，就地修改为 L2 单位向量
        //   n   — 向量维度（Qwen3.5 = 128）
        //   eps — 防止除零的小常数
        void l2norm_inplace_neon(float *x, int n, float eps) {
            // ============================================================
            // Pass 1：平方和累加
            // 使用 4 路独立累加器 s0/s1/s2/s3 隐藏 FMA 指令的 3-4 cycle
            // 延迟链——每条 FMA 依赖上一轮的结果，4 路交错执行让流水线
            // 可以在等待一路结果时发射另一路的指令。
            // ============================================================
            float32x4_t s0 = vdupq_n_f32(0.0f);  // 累加器 0：处理 x[i], x[i+16], ...
            float32x4_t s1 = vdupq_n_f32(0.0f);  // 累加器 1：处理 x[i+4], x[i+20], ...
            float32x4_t s2 = vdupq_n_f32(0.0f);  // 累加器 2：处理 x[i+8], x[i+24], ...
            float32x4_t s3 = vdupq_n_f32(0.0f);  // 累加器 3：处理 x[i+12], x[i+28], ...

            int i = 0;
            // 主循环：每次处理 16 个 float（4 路 × 4 元素/路）
            for (; i + 16 <= n; i += 16) {
                float32x4_t a = vld1q_f32(x + i);       // 加载 x[i:i+4]
                float32x4_t b = vld1q_f32(x + i + 4);   // 加载 x[i+4:i+8]
                float32x4_t c = vld1q_f32(x + i + 8);   // 加载 x[i+8:i+12]
                float32x4_t d = vld1q_f32(x + i + 12);  // 加载 x[i+12:i+16]
                s0 = vfmaq_f32(s0, a, a);  // s0 += a*a（FMA：乘加融合）
                s1 = vfmaq_f32(s1, b, b);  // s1 += b*b
                s2 = vfmaq_f32(s2, c, c);  // s2 += c*c
                s3 = vfmaq_f32(s3, d, d);  // s3 += d*d
            }
            // 中间尾段：4 元素一组（不足 16 但 ≥ 4 的部分）
            for (; i + 4 <= n; i += 4) {
                float32x4_t a = vld1q_f32(x + i);  // 加载 4 个元素
                s0 = vfmaq_f32(s0, a, a);           // 累加到 s0
            }
            // 水平归约：将 4 路累加器的 16 个分量合并为一个标量
            // vaddq_f32(s0,s1) + vaddq_f32(s2,s3) → 两两相加
            // vaddvq_f32(...) → 将 4 个 lane 求和得到标量
            float sumsq = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
            // 标量尾段：不足 4 元素的残余部分
            for (; i < n; ++i) sumsq += x[i] * x[i];

            // ============================================================
            // 计算 inv_norm = 1/sqrt(sumsq + eps)
            // 用 NEON vrsqrte + 2 步 Newton-Raphson 替代标量 sqrt+div，
            // 避免标量/向量域切换带来的流水线停顿。
            //
            // Newton-Raphson 迭代公式（求 1/sqrt(val)）：
            //   est_{n+1} = est_n * (3 - val * est_n²) / 2
            // AArch64 提供 vrsqrtsq_f32(a,b) = (3 - a*b) / 2 直接用，
            // 其中 a = val*est, b = est → 结果为 est * (3 - val*est²) / 2。
            //
            // vrsqrteq_f32 给出 ~12 bits 精度的初始估计。
            // 1 步 Newton → ~24 bits；2 步 → ~48 bits（远超 fp32 的 23 bit 尾数）。
            // ============================================================
            const float val = sumsq + eps;                    // 被开方数
            float32x4_t val_v = vdupq_n_f32(val);             // 广播到 NEON 寄存器
            float32x4_t est = vrsqrteq_f32(val_v);            // 硬件初始估计 ~12 bits
            // Newton 第 1 步：精度 ~12 bits → ~24 bits
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val_v, est), est));
            // Newton 第 2 步：精度 ~24 bits（超过 fp32 尾数精度，已足够精确）
            est = vmulq_f32(est, vrsqrtsq_f32(vmulq_f32(val_v, est), est));
            // 提取 lane 0 作为标量逆范数，广播回 NEON 向量用于 Pass 2
            const float32x4_t vs = vdupq_n_f32(vgetq_lane_f32(est, 0));

            // ============================================================
            // Pass 2：归一化 x[i] *= inv_norm
            // 16 元素/迭代展开，n=128 时仅 8 次循环（vs 4 元素/迭代的 32 次）。
            // 减少循环开销（分支预测、计数器递增、地址计算）。
            // ============================================================
            int j = 0;
            // 主循环：每次处理 16 个 float
            for (; j + 16 <= n; j += 16) {
                vst1q_f32(x + j,      vmulq_f32(vld1q_f32(x + j),      vs));   // x[j:j+4] *= inv_norm
                vst1q_f32(x + j + 4,  vmulq_f32(vld1q_f32(x + j + 4),  vs));   // x[j+4:j+8] *= inv_norm
                vst1q_f32(x + j + 8,  vmulq_f32(vld1q_f32(x + j + 8),  vs));   // x[j+8:j+12] *= inv_norm
                vst1q_f32(x + j + 12, vmulq_f32(vld1q_f32(x + j + 12), vs));   // x[j+12:j+16] *= inv_norm
            }
            // 中间尾段：4 元素一组
            for (; j + 4 <= n; j += 4) {
                vst1q_f32(x + j, vmulq_f32(vld1q_f32(x + j), vs));  // x[j:j+4] *= inv_norm
            }
            // 标量尾段：不足 4 元素的残余部分
            const float inv_norm = vgetq_lane_f32(est, 0);  // 提取标量逆范数
            for (; j < n; ++j) x[j] *= inv_norm;             // 逐元素乘以逆范数
        }
    } // namespace

    // 自注册宏：将此 NEON 实现以 "neon" 名字登记到 l2norm_inplace 分发注册表
    TINYQWEN_L2NORM_INPLACE_VARIANT(l2norm_inplace_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__ || _M_ARM64
