// ============================================================================
// causal_conv1d_update_neon.cpp — Causal Depthwise Conv1D 单步更新的 NEON 实现
// ============================================================================
//
// 本文件实现 GDN（Gated DeltaNet）线性注意力层中的因果深度一维卷积的状态更新。
// 在 Qwen3.5 架构中，每个 GDN 层的输入首先经过一个逐通道的因果卷积，用于
// 捕获局部时序依赖。该卷积是 depthwise 的——每个通道独立计算，不跨通道混合。
//
// 算法（每通道 c 独立）：
//   给定历史状态 state_c = [s0, s1, s2]（最近 3 个输入，旧→新），
//   以及当前输入 x[c] 和卷积权重 weight[c] = [w0, w1, w2, w3]：
//     acc = w0*s0 + w1*s1 + w2*s2 + w3*x[c]   （4 元素点积）
//     out[c] = SiLU(acc)                         （激活函数）
//   然后状态左移一格并推入原始输入（不是 SiLU 后的值，与 HF 对齐）：
//     state_c = [s1, s2, x[c]]
//
// 实际维度（Qwen3.5-0.8B）：
//   dim = 6144（conv_dim），kernel_size = 4（state_len = 3），每层调用 1 次。
//
// 内存布局：
//   conv_state[dim][state_len=3]：通道主序（channel-major），每通道连续存放
//     3 个历史输入。总大小 = 6144 × 3 × 4B = 72KB。
//   weight[dim][kernel_size=4]：通道主序，每通道连续存放 4 个卷积权重。
//     总大小 = 6144 × 4 × 4B = 96KB。
//   x[dim]、out[dim]：当前时间步的输入/输出向量。
//
// NEON 优化策略：
//   ① 跨通道向量化：每次处理 4 个通道（float32x4_t），通道间完全独立。
//      vld3q_f32 一次性加载 4 个通道各自的 3 个 state 元素（硬件 deinterleave）。
//      vld4q_f32 一次性加载 4 个通道各自的 4 个权重（stride=4 天然对应 kernel_size）。
//   ② 双路 8 通道展开：主循环每次处理 8 个通道（2 组 × 4 通道），两组的
//      exp 多项式延迟互相重叠（OoO 调度器可在第 1 组执行 SiLU 时发射
//      第 2 组的 load/FMA），隐藏 7-FMA 延迟链。
//   ③ SiLU 保留 vdivq_f32：Apple Silicon 上 vdivq 延迟仅 ~7 cycles，
//      vrecpe+Newton-Raphson 反而更慢（Cortex-A78 可考虑替换）。
//   ④ 向量化 exp：6 阶泰勒展开 + 整数分离指数部分，相对误差 < 2e-7。
//
// 就地安全：x 和 out 可以指向同一块内存——代码先保存 xc 再写 out[c]，
//   确保后续状态推进使用的是原始输入值。
//
// 数值精度：
//   - exp 多项式逼近 vs std::exp：相对误差 ~2e-7
//   - 单测容差：output ≤ 1e-5，state ≤ 1e-6（state 是精确移位，无近似）
// ============================================================================

#include "dispatch.h"  // 分发层头文件：提供 TINYQWEN_CAUSAL_CONV1D_UPDATE_VARIANT 宏
#include "gdn_ops.h"   // GDN 算子声明：包含 causal_conv1d_update_ref 兜底签名

#if defined(__aarch64__) || defined(_M_ARM64)  // 仅在 AArch64 平台编译

#include <arm_neon.h>  // ARM NEON intrinsic 头文件
#include <cmath>       // std::exp（标量尾段使用）

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部函数不导出符号表
        // ================================================================
        // 向量化 exp（6 阶泰勒展开 + 整数分离指数）
        // ================================================================
        // 算法原理：将 exp(x) 分解为 2^n × exp(f*ln2)，其中 n = round(x/ln2)，
        // |f| ≤ 0.5。2^n 通过 IEEE 754 位操作直接构造（设置指数域），
        // exp(f*ln2) 用 6 阶泰勒多项式逼近。两者相乘得最终结果。
        // 有效范围 [-87, 88]，覆盖 SiLU 所需的全部输入区间。

        constexpr float kLog2e = 1.4426950408889634f;  // log2(e)，用于将自然对数转为以 2 为底
        constexpr float kLn2 = 0.6931471805599453f;    // ln(2)，小数部分的缩放因子

        inline float32x4_t vexpq_f32(float32x4_t x) {
            // y = x * log2(e)：将输入从自然对数域转换到以 2 为底的域
            const float32x4_t y = vmulq_f32(x, vdupq_n_f32(kLog2e));
            // magic number 技巧：12582912.0 = 2^23 * 1.5，加到浮点数上再减去，
            // 利用 IEEE 754 的舍入行为得到最近的整数值（等价于 round-to-nearest-even）
            const float32x4_t magic = vdupq_n_f32(12582912.0f);
            // n_f = round(y)：整数部分（仍为浮点表示）
            const float32x4_t n_f = vsubq_f32(vaddq_f32(y, magic), magic);
            // f = y - n_f：小数部分，|f| ≤ 0.5
            const float32x4_t f = vsubq_f32(y, n_f);

            // 将浮点整数 n_f 转为真正的 int32，用于构造 2^n 的位模式
            int32x4_t n_i = vcvtq_s32_f32(n_f);
            // 钳制到 [-126, 127]：IEEE 754 single 的有效指数范围，防止溢出/下溢
            n_i = vmaxq_s32(n_i, vdupq_n_s32(-126));  // 下界 -126（最小正规格化指数）
            n_i = vminq_s32(n_i, vdupq_n_s32(127));   // 上界 127（最大规格化指数）
            // 构造 2^n：IEEE 754 中 float 的指数域 = n + 127（bias），左移 23 位到指数位置
            const float32x4_t pow2n = vreinterpretq_f32_s32(
                vshlq_n_s32(vaddq_s32(n_i, vdupq_n_s32(127)), 23));

            // 6 阶泰勒展开 exp(f*ln2) ≈ Σ (f*ln2)^k / k!，k=0..6
            // 使用 Horner 法则从高阶到低阶嵌套求值，减少乘法次数
            const float32x4_t t = vmulq_f32(f, vdupq_n_f32(kLn2));  // t = f * ln2
            float32x4_t p = vdupq_n_f32(1.0f / 720.0f);             // c6 = 1/6! = 1/720
            p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), t, p);       // p = c5 + t*p = 1/5! + t/6!
            p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), t, p);        // p = c4 + t*p = 1/4! + t*(...)
            p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), t, p);         // p = c3 + t*p = 1/3! + t*(...)
            p = vfmaq_f32(vdupq_n_f32(0.5f), t, p);                 // p = c2 + t*p = 1/2! + t*(...)
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // p = c1 + t*p = 1 + t*(...)
            p = vfmaq_f32(vdupq_n_f32(1.0f), t, p);                 // p = c0 + t*p = 1 + t*(...)

            // 最终结果 = 2^n × poly(f*ln2)
            return vmulq_f32(pow2n, p);
        }

        // ================================================================
        // 向量化 SiLU（Sigmoid Linear Unit）：silu(g) = g / (1 + exp(-g))
        // Apple Silicon 的 vdivq_f32 延迟仅 ~7 cycles，vrecpe+Newton 在此
        // 平台无优势（Cortex-A78 上 vdivq 延迟 12-15 cycles，可考虑替换）。
        // ================================================================
        inline float32x4_t vsiluq_f32(float32x4_t g) {
            const float32x4_t one = vdupq_n_f32(1.0f);           // 常量 1.0
            float32x4_t e = vexpq_f32(vnegq_f32(g));              // exp(-g)
            return vdivq_f32(g, vaddq_f32(one, e));               // g / (1 + exp(-g))
        }

        // ================================================================
        // causal_conv1d_update_neon — NEON 优化的因果卷积状态更新
        // ================================================================
        // 参数：
        //   x          — 当前时间步的输入向量 [dim]
        //   conv_state — 卷积状态缓冲区 [dim][state_len]，就地更新
        //   weight     — 卷积权重 [dim][kernel_size]
        //   out        — 输出向量 [dim]（可与 x 同址）
        //   dim        — 通道数（Qwen3.5 = 6144）
        //   kernel_size — 卷积核大小（Qwen3.5 = 4）
        void causal_conv1d_update_neon(const float *x, float *conv_state,
                                       const float *weight, float *out,
                                       int dim, int kernel_size) {
            // 非 kernel_size=4 兜底到 ref（Qwen3.5 固定为 4，不应命中此分支）
            if (kernel_size != 4) {
                causal_conv1d_update_ref(x, conv_state, weight, out, dim, kernel_size);
                return;
            }

            const int state_len = 3;  // kernel_size - 1 = 历史状态长度

            // ============================================================
            // 主循环：双路 8 通道展开
            // 每次迭代处理 8 个通道（2 组 × 4 通道/组）。两组独立的 dot+SiLU
            // 管线让乱序执行器可以重叠一组的 exp 多项式延迟与另一组的 load/FMA，
            // 有效隐藏 7-FMA 延迟链。
            // ============================================================
            int c = 0;  // 当前处理的通道索引
            for (; c + 8 <= dim; c += 8) {  // 每次步进 8 个通道
                // --- 第 1 组：通道 c ~ c+3（4 通道并行）---
                // 加载当前输入 x[c..c+3] 到 NEON 寄存器
                float32x4_t xc0 = vld1q_f32(x + c);
                // 计算该组 state 的起始地址（每通道占 state_len=3 个 float）
                float *sp0 = conv_state + c * state_len;
                // vld3q_f32：解交错加载 4 通道 × 3 元素的 state
                // st0.val[0] = 4 个通道的最旧元素 [s0_c, s0_{c+1}, s0_{c+2}, s0_{c+3}]
                // st0.val[1] = 4 个通道的中间元素
                // st0.val[2] = 4 个通道的最新元素
                float32x4x3_t st0 = vld3q_f32(sp0);
                // vld4q_f32：解交错加载 4 通道 × 4 元素的权重
                // wt0.val[i] = 4 个通道的第 i 个权重
                float32x4x4_t wt0 = vld4q_f32(weight + c * kernel_size);

                // 4 元素点积累加：acc = w0*s0 + w1*s1 + w2*s2 + w3*xc
                float32x4_t acc0 = vmulq_f32(wt0.val[0], st0.val[0]);   // acc = w0 * s0
                acc0 = vfmaq_f32(acc0, wt0.val[1], st0.val[1]);          // acc += w1 * s1
                acc0 = vfmaq_f32(acc0, wt0.val[2], st0.val[2]);          // acc += w2 * s2
                acc0 = vfmaq_f32(acc0, wt0.val[3], xc0);                 // acc += w3 * x[c]

                // --- 第 2 组：通道 c+4 ~ c+7（与第 1 组的 SiLU 延迟重叠）---
                float32x4_t xc1 = vld1q_f32(x + c + 4);                  // 加载第 2 组输入
                float *sp1 = conv_state + (c + 4) * state_len;           // 第 2 组 state 地址
                float32x4x3_t st1 = vld3q_f32(sp1);                      // 解交错加载 state
                float32x4x4_t wt1 = vld4q_f32(weight + (c + 4) * kernel_size); // 解交错加载权重

                // 同样的 4 元素点积
                float32x4_t acc1 = vmulq_f32(wt1.val[0], st1.val[0]);   // acc1 = w0 * s0
                acc1 = vfmaq_f32(acc1, wt1.val[1], st1.val[1]);          // acc1 += w1 * s1
                acc1 = vfmaq_f32(acc1, wt1.val[2], st1.val[2]);          // acc1 += w2 * s2
                acc1 = vfmaq_f32(acc1, wt1.val[3], xc1);                 // acc1 += w3 * x[c+4]

                // SiLU 激活：两组独立发射，OoO 调度器可重叠 exp 多项式延迟
                vst1q_f32(out + c,     vsiluq_f32(acc0));   // 写出第 1 组结果
                vst1q_f32(out + c + 4, vsiluq_f32(acc1));   // 写出第 2 组结果

                // 状态左移：st[0]=st[1], st[1]=st[2], st[2]=原始输入
                // 注意：这里存入的是原始输入 xc（不是 SiLU 后的值），与 HF 实现对齐
                st0.val[0] = st0.val[1];   // 最旧 ← 中间
                st0.val[1] = st0.val[2];   // 中间 ← 最新
                st0.val[2] = xc0;          // 最新 ← 当前原始输入
                vst3q_f32(sp0, st0);       // 交错写回 state（4 通道 × 3 元素）

                st1.val[0] = st1.val[1];   // 同上，第 2 组状态左移
                st1.val[1] = st1.val[2];
                st1.val[2] = xc1;
                vst3q_f32(sp1, st1);       // 交错写回
            }

            // ============================================================
            // 单路 4 通道处理剩余部分（dim 非 8 整除但 ≥ 4 的尾部）
            // ============================================================
            for (; c + 4 <= dim; c += 4) {
                float32x4_t xc = vld1q_f32(x + c);                       // 加载 4 通道输入
                float *sp = conv_state + c * state_len;                   // state 起始地址
                float32x4x3_t st = vld3q_f32(sp);                        // 解交错加载 state
                float32x4x4_t wt = vld4q_f32(weight + c * kernel_size);  // 解交错加载权重

                // 4 元素点积
                float32x4_t acc = vmulq_f32(wt.val[0], st.val[0]);      // acc = w0 * s0
                acc = vfmaq_f32(acc, wt.val[1], st.val[1]);              // acc += w1 * s1
                acc = vfmaq_f32(acc, wt.val[2], st.val[2]);              // acc += w2 * s2
                acc = vfmaq_f32(acc, wt.val[3], xc);                     // acc += w3 * x

                vst1q_f32(out + c, vsiluq_f32(acc));                    // SiLU + 写出

                // 状态左移 + 推入原始输入
                st.val[0] = st.val[1];   // 最旧 ← 中间
                st.val[1] = st.val[2];   // 中间 ← 最新
                st.val[2] = xc;          // 最新 ← 当前原始输入
                vst3q_f32(sp, st);       // 交错写回
            }

            // ============================================================
            // 标量尾段（dim 非 4 整除时，逐通道处理）
            // ============================================================
            for (; c < dim; ++c) {
                const float xc = x[c];  // 保存原始输入（x 与 out 可能同址）
                // 计算该通道 state 的起始地址
                float *st = conv_state + static_cast<size_t>(c) * state_len;
                // 计算该通道权重的起始地址
                const float *w = weight + static_cast<size_t>(c) * kernel_size;

                // 标量点积：state 中的 3 个历史值 + 当前输入
                float acc = 0.0f;
                for (int i = 0; i < state_len; ++i) acc += w[i] * st[i];  // Σ w[i]*state[i]
                acc += w[state_len] * xc;                                  // + w[3]*x[c]

                // 标量 SiLU：exp + div
                float e = std::exp(-acc);          // exp(-acc)
                out[c] = acc / (1.0f + e);         // silu(acc) = acc / (1 + exp(-acc))

                // 状态左移一格
                for (int i = 0; i < state_len - 1; ++i) st[i] = st[i + 1];  // s[0]=s[1], s[1]=s[2]
                st[state_len - 1] = xc;  // 最新槽位 ← 原始输入
            }
        }
    } // namespace

    // 自注册宏：将此 NEON 实现以 "neon" 名字登记到 causal_conv1d_update 分发注册表
    TINYQWEN_CAUSAL_CONV1D_UPDATE_VARIANT(causal_conv1d_update_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__ || _M_ARM64
