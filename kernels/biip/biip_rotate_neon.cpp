// ============================================================================
// biip_rotate_neon.cpp — BiIP 激活旋转的 NEON SIMD 优化版（aarch64）
// ============================================================================
// 与 biip_rotate_activation_ref 计算完全相同的三步：
//   1. y[j] = x[j] / scale[j]      （scale == nullptr 时跳过）
//   2. y[j] *= sign[j]
//   3. 每 block_size 个元素做归一化 Walsh-Hadamard（butterfly），末乘 1/sqrt(bs)
//
// 为什么值得向量化：旋转量化模型每层 7 个子层各旋转一次且**不能融合**
// （每个子层有各自的 sign/scale），Qwen3-0.6B 每 token 就是 196 次调用。
// 标量版实测只跑到 4.3 GB/s（机器天花板 114 GB/s），且第 1 步是逐元素除法。
//
// **逐位一致性**：本实现不改变任何运算的顺序或结合方式——
//   - 除法用 vdivq_f32（IEEE 精确，不用 vrecpe 近似），与标量 `/` 同结果；
//   - butterfly 的每个 (a+c, a-c) 仍是各自独立的一次 fp32 加/减，只是 4 个
//     元素同时算，浮点加法逐元素确定，不存在重结合；
//   - 末尾 1/sqrt(bs) 仍是 butterfly 之后单独一遍乘（不折进第 1 步——
//     block_size=128 时 1/sqrt(128) 不是 2 的幂，提前折会改变舍入）。
// 因此与 ref 的差应恒为 0，由 tests/test_biip_rotate.cpp 逐变体校验。
//
// butterfly 的向量化分三段（block_size 是 2 的幂）：
//   - h >= 4：blk[j..j+3] 与 blk[j+h..j+h+3] 天然 4 元素对齐，直接 vaddq/vsubq；
//   - h == 2：配对在一个 4 向量内的低/高 64 位半之间，用 vget_low/high + vcombine；
//   - h == 1：配对是相邻元素，用 vuzp1q/vuzp2q 分出偶/奇再 vzip1q/vzip2q 交织回去。
// ============================================================================

#include "dispatch.h"
#include "ref_ops.h"

#if defined(__aarch64__)

#include <arm_neon.h>
#include <cmath>

namespace tinyqwen {
  namespace {

    // 一块 block_size 元素的归一化 Walsh-Hadamard（原地，NEON）
    void fwht_block_neon(float *blk, int block_size, float inv_sqrt) {
      // ---- 阶段 A：h == 1，配对 (0,1) (2,3) ... 落在同一个 4 向量内 ----
      // [a0,a1,a2,a3] -> [a0+a1, a0-a1, a2+a3, a2-a3]
      for (int i = 0; i < block_size; i += 4) {
        const float32x4_t v = vld1q_f32(blk + i);
        const float32x4_t even = vuzp1q_f32(v, v);   // [a0,a2,a0,a2]
        const float32x4_t odd = vuzp2q_f32(v, v);    // [a1,a3,a1,a3]
        const float32x4_t s = vaddq_f32(even, odd);  // [a0+a1, a2+a3, ...]
        const float32x4_t d = vsubq_f32(even, odd);  // [a0-a1, a2-a3, ...]
        vst1q_f32(blk + i, vzip1q_f32(s, d));        // [s0,d0,s1,d1]
      }
      if (block_size == 2) {  // 只有 h=1 一级（dim 被 2 整除的退化块大小）
        for (int j = 0; j < block_size; ++j) blk[j] *= inv_sqrt;
        return;
      }

      // ---- 阶段 B：h == 2，配对 (0,2) (1,3) 落在同一个 4 向量的两个 64 位半 ----
      // [a0,a1,a2,a3] -> [a0+a2, a1+a3, a0-a2, a1-a3]
      for (int i = 0; i < block_size; i += 4) {
        const float32x4_t v = vld1q_f32(blk + i);
        const float32x2_t lo = vget_low_f32(v);   // [a0,a1]
        const float32x2_t hi = vget_high_f32(v);  // [a2,a3]
        vst1q_f32(blk + i, vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi)));
      }

      // ---- 阶段 C：h >= 4，两侧各自 4 元素对齐，直接向量加减 ----
      for (int h = 4; h < block_size; h <<= 1) {
        for (int i = 0; i < block_size; i += 2 * h) {
          float *lo = blk + i;
          float *hi = blk + i + h;
          for (int j = 0; j < h; j += 4) {
            const float32x4_t a = vld1q_f32(lo + j);
            const float32x4_t c = vld1q_f32(hi + j);
            vst1q_f32(lo + j, vaddq_f32(a, c));
            vst1q_f32(hi + j, vsubq_f32(a, c));
          }
        }
      }

      // ---- 归一化：整块乘 1/sqrt(block_size) ----
      const float32x4_t s = vdupq_n_f32(inv_sqrt);
      for (int j = 0; j < block_size; j += 4) {
        vst1q_f32(blk + j, vmulq_f32(vld1q_f32(blk + j), s));
      }
    }

    void biip_rotate_activation_neon(const float *x, float *y, int dim,
                                     const float *scale, const float *sign, int block_size) {
      // block_size < 4 时向量化拿不到收益且要处理大量边界，直接交给 ref
      if (block_size < 4 || (block_size & (block_size - 1)) != 0 || dim % block_size != 0) {
        biip_rotate_activation_ref(x, y, dim, scale, sign, block_size);
        return;
      }

      // ---- 步骤 1+2：逐元素除 scale、乘 sign（dim 必被 block_size 整除，故必是 4 的倍数）----
      if (scale) {
        for (int j = 0; j < dim; j += 4) {
          const float32x4_t q = vdivq_f32(vld1q_f32(x + j), vld1q_f32(scale + j));
          vst1q_f32(y + j, vmulq_f32(q, vld1q_f32(sign + j)));
        }
      } else {
        for (int j = 0; j < dim; j += 4) {
          vst1q_f32(y + j, vmulq_f32(vld1q_f32(x + j), vld1q_f32(sign + j)));
        }
      }

      // ---- 步骤 3：逐块 butterfly ----
      const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(block_size));
      const int num_blocks = dim / block_size;
      for (int b = 0; b < num_blocks; ++b) {
        fwht_block_neon(y + static_cast<size_t>(b) * block_size, block_size, inv_sqrt);
      }
    }

  } // namespace

  // 自注册进 biip 旋转注册表，名称 "neon"（与 --ops-impl neon 共用名字）。
  TINYQWEN_BIIP_ROTATE_VARIANT(biip_rotate_activation_neon, "neon");
} // namespace tinyqwen

#endif // __aarch64__
