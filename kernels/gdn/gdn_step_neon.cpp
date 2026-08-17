// ============================================================================
// GDN 递归步（Gated Delta Rule）的 NEON 优化实现
// ============================================================================
//
// 算法概述（单 head、单 token）：
//   1. S *= exp(g)                   衰减门控
//   2. kv_mem = S^T @ k              用衰减后的状态读出旧值
//   3. delta = beta * (v - kv_mem)   delta rule 纠错量
//   4. S += outer(k, delta)          状态写入
//   5. o = S^T @ q                   输出读取
//
// 实际维度（Qwen3.5-0.8B）：
//   qk_dim = 128, v_dim = 128, S = [128×128] = 64KB
//   每层调用 16 次（16 个 GDN head），是 GDN 层的主要耗时算子。
//
// 优化策略：
//   ① 融合 Pass 2+3：步骤 4 和 5 合并为一次 S 遍历，减少 64KB 内存读取
//      （从 3 遍 → 2 遍，节省 ~33% 内存流量）
//   ② 4× 循环展开：内层 d 循环每次处理 16 个 float（4 个 float32x4_t），
//      减少循环开销、改善乱序执行调度
//   ③ 软件预取：提前 2 行预取 S 的下一行，隐藏 L2 未命中延迟
//   ④ 分块策略：v_dim=128 分为 2 块（chunk=64 = 16 个累加器），
//      留出寄存器给 vkj/vqj/vdecay 等临时值（AArch64 共 32 个 NEON 寄存器）
//
// 数值精度：
//   - Pass 1/2 的乘加舍入：vs ref 差异 ~1e-6（S 矩阵）
//   - Pass 3 用 float 累加 128 维点积（ref 用 double）：o 误差 ~1e-4
//   - 单测容差：o ≤ 1e-4，S ≤ 1e-5
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cmath>

namespace tinyqwen {
    namespace {
        void gdn_step_neon(float *S, const float *q, const float *k, const float *v,
                           float g, float beta, float *o, int qk_dim, int v_dim) {
            const float decay = std::exp(g);
            const float32x4_t vdecay = vdupq_n_f32(decay);
            const float32x4_t vbeta = vdupq_n_f32(beta);

            // 分块大小 = 64（16 个 float32x4_t 累加器）。
            // 寄存器预算：16 累加器 + vdecay + vkj + 4 临时 = 22，留 10 个给编译器。
            constexpr int kChunk = 64;

            for (int d0 = 0; d0 < v_dim; d0 += kChunk) {
                const int dend = (d0 + kChunk <= v_dim) ? d0 + kChunk : v_dim;
                const int chunk_len = dend - d0;

                // ================================================================
                // Pass 1：S[j][d] *= decay; kv_mem[d] = Σ_j S[j][d] * k[j]
                // 同时完成衰减和 key-value 记忆读取（不可与后续融合——
                // delta 依赖完整的 kv_mem，即所有 j 行的贡献）。
                // ================================================================
                float32x4_t kv_mem[16];
                for (int i = 0; i < 16; ++i) kv_mem[i] = vdupq_n_f32(0.0f);

                for (int j = 0; j < qk_dim; ++j) {
                    float *Srow = S + static_cast<size_t>(j) * v_dim + d0;

                    // 预取 2 行后的数据（隐藏 L1 miss → L2 的延迟）
                    if (j + 2 < qk_dim) {
                        __builtin_prefetch(S + static_cast<size_t>(j + 2) * v_dim + d0, 1, 3);
                    }

                    const float32x4_t vkj = vdupq_n_f32(k[j]);

                    // 4× 展开：每次处理 16 个 float
                    int d = 0;
                    for (; d + 16 <= chunk_len; d += 16) {
                        float32x4_t s0 = vmulq_f32(vld1q_f32(Srow + d),      vdecay);
                        float32x4_t s1 = vmulq_f32(vld1q_f32(Srow + d + 4),  vdecay);
                        float32x4_t s2 = vmulq_f32(vld1q_f32(Srow + d + 8),  vdecay);
                        float32x4_t s3 = vmulq_f32(vld1q_f32(Srow + d + 12), vdecay);

                        vst1q_f32(Srow + d,      s0);
                        vst1q_f32(Srow + d + 4,  s1);
                        vst1q_f32(Srow + d + 8,  s2);
                        vst1q_f32(Srow + d + 12, s3);

                        kv_mem[d / 4]     = vfmaq_f32(kv_mem[d / 4],     s0, vkj);
                        kv_mem[d / 4 + 1] = vfmaq_f32(kv_mem[d / 4 + 1], s1, vkj);
                        kv_mem[d / 4 + 2] = vfmaq_f32(kv_mem[d / 4 + 2], s2, vkj);
                        kv_mem[d / 4 + 3] = vfmaq_f32(kv_mem[d / 4 + 3], s3, vkj);
                    }
                    // 尾段处理（chunk_len 非 16 倍数时）
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t s = vmulq_f32(vld1q_f32(Srow + d), vdecay);
                        vst1q_f32(Srow + d, s);
                        kv_mem[d / 4] = vfmaq_f32(kv_mem[d / 4], s, vkj);
                    }
                    for (; d < chunk_len; ++d) {
                        Srow[d] *= decay;
                    }
                }

                // ================================================================
                // 计算 delta = beta * (v - kv_mem)
                // delta 对整个 chunk 只计算一次，供下面融合遍使用。
                // ================================================================
                float32x4_t delta[16];
                {
                    int d = 0;
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t vv = vld1q_f32(v + d0 + d);
                        delta[d / 4] = vmulq_f32(vbeta, vsubq_f32(vv, kv_mem[d / 4]));
                    }
                }

                // ================================================================
                // 融合 Pass 2+3：S[j][d] += k[j]*delta[d]; o[d] += S[j][d]*q[j]
                //
                // 关键优化：原来 Pass 2 写完 S 后，Pass 3 再读一遍 S 做点积。
                // 融合后在同一次遍历中完成写入和输出累加，省去 64KB 的重复读取。
                // ================================================================
                float32x4_t o_acc[16];
                for (int i = 0; i < 16; ++i) o_acc[i] = vdupq_n_f32(0.0f);

                for (int j = 0; j < qk_dim; ++j) {
                    float *Srow = S + static_cast<size_t>(j) * v_dim + d0;

                    // 预取 2 行后的数据
                    if (j + 2 < qk_dim) {
                        __builtin_prefetch(S + static_cast<size_t>(j + 2) * v_dim + d0, 1, 3);
                    }

                    const float32x4_t vkj = vdupq_n_f32(k[j]);
                    const float32x4_t vqj = vdupq_n_f32(q[j]);

                    // 4× 展开：更新 S 并同时累加输出
                    int d = 0;
                    for (; d + 16 <= chunk_len; d += 16) {
                        // 加载 + 更新 S
                        float32x4_t s0 = vfmaq_f32(vld1q_f32(Srow + d),      vkj, delta[d / 4]);
                        float32x4_t s1 = vfmaq_f32(vld1q_f32(Srow + d + 4),  vkj, delta[d / 4 + 1]);
                        float32x4_t s2 = vfmaq_f32(vld1q_f32(Srow + d + 8),  vkj, delta[d / 4 + 2]);
                        float32x4_t s3 = vfmaq_f32(vld1q_f32(Srow + d + 12), vkj, delta[d / 4 + 3]);

                        vst1q_f32(Srow + d,      s0);
                        vst1q_f32(Srow + d + 4,  s1);
                        vst1q_f32(Srow + d + 8,  s2);
                        vst1q_f32(Srow + d + 12, s3);

                        // 累加输出 o[d] += S[j][d] * q[j]
                        o_acc[d / 4]     = vfmaq_f32(o_acc[d / 4],     s0, vqj);
                        o_acc[d / 4 + 1] = vfmaq_f32(o_acc[d / 4 + 1], s1, vqj);
                        o_acc[d / 4 + 2] = vfmaq_f32(o_acc[d / 4 + 2], s2, vqj);
                        o_acc[d / 4 + 3] = vfmaq_f32(o_acc[d / 4 + 3], s3, vqj);
                    }
                    // 尾段
                    for (; d + 4 <= chunk_len; d += 4) {
                        float32x4_t s = vfmaq_f32(vld1q_f32(Srow + d), vkj, delta[d / 4]);
                        vst1q_f32(Srow + d, s);
                        o_acc[d / 4] = vfmaq_f32(o_acc[d / 4], s, vqj);
                    }
                    for (; d < chunk_len; ++d) {
                        // 标量尾段：从 delta 向量数组中提取对应标量值
                        float delta_buf[4];
                        vst1q_f32(delta_buf, delta[d / 4]);
                        Srow[d] += k[j] * delta_buf[d % 4];
                    }
                }

                // 写出输出向量
                {
                    int d = 0;
                    for (; d + 4 <= chunk_len; d += 4) {
                        vst1q_f32(o + d0 + d, o_acc[d / 4]);
                    }
                    // 尾段标量回退
                    for (; d < chunk_len; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j < qk_dim; ++j) {
                            acc += S[static_cast<size_t>(j) * v_dim + d0 + d] * q[j];
                        }
                        o[d0 + d] = acc;
                    }
                }
            }
        }
    } // namespace

    TINYQWEN_GDN_STEP_VARIANT(gdn_step_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__
