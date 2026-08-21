// ============================================================================
// gdn_step_neon.cpp — Gated Delta Rule 单步递归的 NEON 优化实现
// ============================================================================
//
// 本文件实现 GDN（Gated DeltaNet）线性注意力层的核心递归步骤——
// Gated Delta Rule 状态更新。这是整个 GDN 层中计算量最大的算子，
// 占 GDN 非投影算子耗时的 91%，是优化的首要目标。
//
// 算法概述（单 head、单 token）：
//   1. S *= exp(g)                   — 衰减门控：遗忘旧信息
//   2. kv_mem = S^T @ k              — 用衰减后的状态读出旧 key-value 记忆
//   3. delta = beta * (v - kv_mem)   — delta rule 纠错量
//   4. S += outer(k, delta)          — 将纠错量写入状态矩阵
//   5. o = S^T @ q                   — 从更新后的状态读出输出
//
// 实际维度（Qwen3.5-0.8B）：
//   qk_dim = 128, v_dim = 128, S = [128×128] = 64KB
//   每层调用 16 次（16 个 GDN head），是 GDN 层的主要耗时算子。
//
// 内存布局：
//   S[qk_dim][v_dim]：行主序，S[j][d] = S[j * v_dim + d]。
//   64KB 恰好可放入 Apple M 系列的 128KB L1D cache。
//
// 优化策略：
//   ① 遍融合（Pass Fusion）：原始 ref 实现需要 3 遍遍历 S（64KB × 3 = 192KB
//      内存流量）。NEON 版将 Pass 2（outer product 更新）和 Pass 3（输出读取）
//      融合为一次遍历，节省 ~33% 内存搬运（从 3 遍 → 2 遍）。
//      注意：Pass 1 不可与后续融合——delta 依赖完整的 kv_mem（所有 j 行的贡献）。
//   ② 分块策略（Chunking）：v_dim=128 分为 2 块（chunk=64 = 16 个 float32x4_t
//      累加器）。AArch64 有 32 个 NEON 寄存器，16 个用于累加器后还剩 16 个
//      给 vkj/vqj/vdecay/临时值/编译器 spill。
//   ③ 4× 循环展开：内层 d 循环每次处理 16 个 float（4 个 float32x4_t），
//      减少循环开销（分支预测、计数器递增）、改善乱序执行调度窗口。
//   ④ 软件预取：__builtin_prefetch 提前 2 行预取 S 的下一行数据，
//      隐藏 L1 miss → L2 的访问延迟（~10-20 cycles）。
//
// 数值精度：
//   - Pass 1/2 的乘加舍入：vs ref 差异 ~1e-6（S 矩阵元素级）
//   - Pass 3 用 float 累加 128 维点积（ref 用 double）：o 误差 ~1e-4
//   - 单测容差：output ≤ 1e-4，state ≤ 1e-5
// ============================================================================

#include "dispatch.h"  // 分发层头文件：提供 TINYQWEN_GDN_STEP_VARIANT 宏

#if defined(__aarch64__) || defined(_M_ARM64)  // 仅在 AArch64 平台编译

#include <arm_neon.h>  // ARM NEON intrinsic 头文件
#include <cmath>       // std::exp（计算衰减因子）

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部函数不导出符号表
        // ================================================================
        // gdn_step_neon — NEON 优化的 Gated Delta Rule 单步递归
        // ================================================================
        // 参数：
        //   S       — 状态矩阵 [qk_dim × v_dim]，就地更新
        //   q       — query 向量 [qk_dim]
        //   k       — key 向量 [qk_dim]
        //   v       — value 向量 [v_dim]
        //   g       — 门控标量（log-space），decay = exp(g)
        //   beta    — delta rule 缩放因子
        //   o       — 输出向量 [v_dim]
        //   qk_dim  — query/key 维度（Qwen3.5 = 128）
        //   v_dim   — value 维度（Qwen3.5 = 128）
        void gdn_step_neon(float *S, const float *q, const float *k, const float *v,
                           float g, float beta, float *o, int qk_dim, int v_dim) {
            const float decay = std::exp(g);           // log-space → 乘法衰减因子
            const float32x4_t vdecay = vdupq_n_f32(decay);  // 广播到 NEON 寄存器
            const float32x4_t vbeta = vdupq_n_f32(beta);    // beta 广播到 NEON 寄存器

            // 分块大小 = 64（16 个 float32x4_t 累加器）。
            // 寄存器预算分析：16 累加器 + vdecay + vkj + 4 临时 ≈ 22，
            // AArch64 共 32 个 NEON 寄存器，留 10 个给编译器分配/spill。
            constexpr int kChunk = 64;

            // 外层循环：按 chunk 遍历 v_dim 维度
            for (int d0 = 0; d0 < v_dim; d0 += kChunk) {
                // 计算当前 chunk 的实际结束位置（最后一个 chunk 可能不满 64）
                const int dend = (d0 + kChunk <= v_dim) ? d0 + kChunk : v_dim;
                const int chunk_len = dend - d0;  // 当前 chunk 的实际长度

                // ================================================================
                // Pass 1：S[j][d] *= decay; kv_mem[d] = Σ_j S[j][d] * k[j]
                // 同时完成衰减和 key-value 记忆读取。
                // 此遍不可与后续融合——delta 依赖完整的 kv_mem，即需要所有
                // qk_dim 行的 S[j][d]*k[j] 之和才能计算 delta = beta*(v-kv_mem)。
                // ================================================================

                // kv_mem 累加器数组：16 个 float32x4_t = 64 个 float
                float32x4_t kv_mem[16];
                for (int i = 0; i < 16; ++i) kv_mem[i] = vdupq_n_f32(0.0f);  // 初始化为零

                // 遍历所有 qk_dim 行
                for (int j = 0; j < qk_dim; ++j) {
                    // S 的第 j 行在当前 chunk 范围内的起始地址
                    float *Srow = S + static_cast<size_t>(j) * v_dim + d0;

                    // 软件预取：提前 2 行预取 S 的数据到 L1 cache
                    // 参数：(地址, 读意图=1, 时间局部性=3 即保留在 L1/L2)
                    if (j + 2 < qk_dim) {
                        __builtin_prefetch(S + static_cast<size_t>(j + 2) * v_dim + d0, 1, 3);
                    }

                    // 将 k[j] 广播到 NEON 寄存器（同一行所有 d 共享同一个 k[j]）
                    const float32x4_t vkj = vdupq_n_f32(k[j]);

                    // ---- 4× 展开的主循环：每次处理 16 个 float ----
                    int d = 0;
                    for (; d + 16 <= chunk_len; d += 16) {
                        // 加载 S[j][d..d+15] 并乘以衰减因子
                        float32x4_t s0 = vmulq_f32(vld1q_f32(Srow + d),      vdecay);  // S[j][d:d+4] *= decay
                        float32x4_t s1 = vmulq_f32(vld1q_f32(Srow + d + 4),  vdecay);  // S[j][d+4:d+8] *= decay
                        float32x4_t s2 = vmulq_f32(vld1q_f32(Srow + d + 8),  vdecay);  // S[j][d+8:d+12] *= decay
                        float32x4_t s3 = vmulq_f32(vld1q_f32(Srow + d + 12), vdecay);  // S[j][d+12:d+16] *= decay

                        // 写回衰减后的 S（就地更新）
                        vst1q_f32(Srow + d,      s0);   // 写回 S[j][d:d+4]
                        vst1q_f32(Srow + d + 4,  s1);   // 写回 S[j][d+4:d+8]
                        vst1q_f32(Srow + d + 8,  s2);   // 写回 S[j][d+8:d+12]
                        vst1q_f32(Srow + d + 12, s3);   // 写回 S[j][d+12:d+16]

                        // kv_mem[d'] += S[j][d'] * k[j]（FMA 累加）
                        kv_mem[d / 4]     = vfmaq_f32(kv_mem[d / 4],     s0, vkj);  // kv_mem[d:d+4] += s0 * k[j]
                        kv_mem[d / 4 + 1] = vfmaq_f32(kv_mem[d / 4 + 1], s1, vkj);  // kv_mem[d+4:d+8] += s1 * k[j]
                        kv_mem[d / 4 + 2] = vfmaq_f32(kv_mem[d / 4 + 2], s2, vkj);  // kv_mem[d+8:d+12] += s2 * k[j]
                        kv_mem[d / 4 + 3] = vfmaq_f32(kv_mem[d / 4 + 3], s3, vkj);  // kv_mem[d+12:d+16] += s3 * k[j]
                    }
                    // 尾段处理（chunk_len 非 16 倍数时，4 元素一组）
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t s = vmulq_f32(vld1q_f32(Srow + d), vdecay);  // 加载 + 衰减
                        vst1q_f32(Srow + d, s);                                    // 写回
                        kv_mem[d / 4] = vfmaq_f32(kv_mem[d / 4], s, vkj);         // 累加 kv_mem
                    }
                    // 标量尾段（chunk_len 非 4 倍数时）
                    for (; d < chunk_len; ++d) {
                        Srow[d] *= decay;  // 逐元素衰减
                    }
                }

                // ================================================================
                // 计算 delta = beta * (v - kv_mem)
                // delta 对整个 chunk 只计算一次，供下面融合遍使用。
                // delta[d] 表示 value 空间中当前输入 v 与状态记忆 kv_mem 的差异，
                // 乘以 beta 控制更新强度。
                // ================================================================
                float32x4_t delta[16];  // delta 向量（NEON 格式）
                {
                    int d = 0;
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t vv = vld1q_f32(v + d0 + d);                   // 加载 v[d0+d:d0+d+4]
                        delta[d / 4] = vmulq_f32(vbeta, vsubq_f32(vv, kv_mem[d / 4]));  // beta * (v - kv_mem)
                    }
                }

                // ================================================================
                // 融合 Pass 2+3：S[j][d] += k[j]*delta[d]; o[d] += S[j][d]*q[j]
                //
                // 关键优化：原始 ref 实现中 Pass 2 写完 S 后，Pass 3 再读一遍 S
                // 做点积（两次各 64KB 内存流量）。融合后在同一次遍历中同时完成
                // 状态写入和输出累加，省去一整遍 64KB 的重复读取。
                // ================================================================

                // 输出累加器数组：16 个 float32x4_t = 64 个 float
                float32x4_t o_acc[16];
                for (int i = 0; i < 16; ++i) o_acc[i] = vdupq_n_f32(0.0f);  // 初始化为零

                // 遍历所有 qk_dim 行（与 Pass 1 相同的遍历模式）
                for (int j = 0; j < qk_dim; ++j) {
                    float *Srow = S + static_cast<size_t>(j) * v_dim + d0;  // S 第 j 行 chunk 起始

                    // 软件预取：提前 2 行预取
                    if (j + 2 < qk_dim) {
                        __builtin_prefetch(S + static_cast<size_t>(j + 2) * v_dim + d0, 1, 3);
                    }

                    const float32x4_t vkj = vdupq_n_f32(k[j]);  // k[j] 广播
                    const float32x4_t vqj = vdupq_n_f32(q[j]);  // q[j] 广播

                    // ---- 4× 展开：更新 S 并同时累加输出 ----
                    int d = 0;
                    for (; d + 16 <= chunk_len; d += 16) {
                        // 加载 S[j][d..d+15]，加上 k[j]*delta[d..d+15]（outer product 更新）
                        float32x4_t s0 = vfmaq_f32(vld1q_f32(Srow + d),      vkj, delta[d / 4]);      // S += k*delta
                        float32x4_t s1 = vfmaq_f32(vld1q_f32(Srow + d + 4),  vkj, delta[d / 4 + 1]);  // S += k*delta
                        float32x4_t s2 = vfmaq_f32(vld1q_f32(Srow + d + 8),  vkj, delta[d / 4 + 2]);  // S += k*delta
                        float32x4_t s3 = vfmaq_f32(vld1q_f32(Srow + d + 12), vkj, delta[d / 4 + 3]);  // S += k*delta

                        // 写回更新后的 S
                        vst1q_f32(Srow + d,      s0);   // 写回 S[j][d:d+4]
                        vst1q_f32(Srow + d + 4,  s1);   // 写回 S[j][d+4:d+8]
                        vst1q_f32(Srow + d + 8,  s2);   // 写回 S[j][d+8:d+12]
                        vst1q_f32(Srow + d + 12, s3);   // 写回 S[j][d+12:d+16]

                        // 同时累加输出：o_acc[d'] += S[j][d'] * q[j]
                        o_acc[d / 4]     = vfmaq_f32(o_acc[d / 4],     s0, vqj);  // o[d:d+4] += s0 * q[j]
                        o_acc[d / 4 + 1] = vfmaq_f32(o_acc[d / 4 + 1], s1, vqj);  // o[d+4:d+8] += s1 * q[j]
                        o_acc[d / 4 + 2] = vfmaq_f32(o_acc[d / 4 + 2], s2, vqj);  // o[d+8:d+12] += s2 * q[j]
                        o_acc[d / 4 + 3] = vfmaq_f32(o_acc[d / 4 + 3], s3, vqj);  // o[d+12:d+16] += s3 * q[j]
                    }
                    // 尾段（4 元素一组）
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t s = vfmaq_f32(vld1q_f32(Srow + d), vkj, delta[d / 4]);  // S += k*delta
                        vst1q_f32(Srow + d, s);                                              // 写回 S
                        o_acc[d / 4] = vfmaq_f32(o_acc[d / 4], s, vqj);                      // o += S*q
                    }
                    // 标量尾段（chunk_len 非 4 倍数时）
                    for (; d < chunk_len; ++d) {
                        // 从 delta 向量数组中提取对应标量值
                        float delta_buf[4];                                  // 临时缓冲区
                        vst1q_f32(delta_buf, delta[d / 4]);                 // 将 NEON 向量存到栈上
                        Srow[d] += k[j] * delta_buf[d % 4];                // 标量 outer product 更新
                    }
                }

                // ================================================================
                // 写出输出向量 o[d0..dend]
                // ================================================================
                {
                    int d = 0;
                    // NEON 部分：4 元素一组写出
                    for (; d + 4 <= chunk_len; d += 4) {
                        vst1q_f32(o + d0 + d, o_acc[d / 4]);  // 写出 o[d0+d:d0+d+4]
                    }
                    // 标量尾段回退：对于不足 4 元素的尾部，重新从 S 计算
                    // （因为上面的标量尾段没有累加到 o_acc 中）
                    for (; d < chunk_len; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j < qk_dim; ++j) {
                            acc += S[static_cast<size_t>(j) * v_dim + d0 + d] * q[j];
                        }
                        o[d0 + d] = acc;  // 标量点积结果
                    }
                }
            }  // 外层 chunk 循环结束
        }
    } // namespace

    // 自注册宏：将此 NEON 实现以 "neon" 名字登记到 gdn_step 分发注册表
    TINYQWEN_GDN_STEP_VARIANT(gdn_step_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__ || _M_ARM64
