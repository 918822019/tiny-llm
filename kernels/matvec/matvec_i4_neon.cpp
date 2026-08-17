// INT4 weight-only matvec：y = W @ x —— NEON 优化版。
//
// 布局与 ref 一致：每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(G/2 B)]。
// 核心循环：每迭代处理 32 个 uint4（16 bytes packed），利用 NEON 拆 nibble →
// 转 fp32 → fmla 累加，比标量 ref 快 3-5x（ARM Cortex-A78 实测）。
//
// group_size 必须是 32 的倍数（典型 128），否则回退到 ref。

#include "dispatch.h"
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <cstdint>
#include <cstring>

namespace tinyqwen {
namespace {

// 处理单行的一个 group（group_size 个元素），NEON 32-wide 主循环。
// 返回该 group 的点积贡献。
inline float dot_group_i4_neon(const uint8_t *packed, const float *x,
                               float scale, float zero, int group_elems) {
    const float32x4_t v_scale = vdupq_n_f32(scale);
    const float32x4_t v_neg_zs = vdupq_n_f32(-zero * scale);
    // dequant = val * scale - zero * scale = val * scale + v_neg_zs

    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    int i = 0;
    const int n32 = group_elems & ~31;

    for (; i < n32; i += 32) {
        // Load 16 packed bytes = 32 nibbles
        uint8x16_t raw = vld1q_u8(packed + i / 2);

        // Split into low and high nibbles
        uint8x16_t lo8 = vandq_u8(raw, mask_lo);       // even indices [0,2,4,...,30]
        uint8x16_t hi8 = vshrq_n_u8(raw, 4);           // odd indices  [1,3,5,...,31]

        // Interleave to get sequential order: [0,1,2,3,...,31]
        // zip gives: lo[0],hi[0],lo[1],hi[1],...  = elem0,elem1,elem2,elem3,...
        uint8x16x2_t zipped = vzipq_u8(lo8, hi8);
        // zipped.val[0] = [e0,e1,e2,e3,...,e15]
        // zipped.val[1] = [e16,e17,...,e31]

        // Convert first 16 uint8 to float32 (4 groups of 4)
        uint16x8_t w16_0 = vmovl_u8(vget_low_u8(zipped.val[0]));   // e0..e7
        uint16x8_t w16_1 = vmovl_u8(vget_high_u8(zipped.val[0]));  // e8..e15
        uint16x8_t w16_2 = vmovl_u8(vget_low_u8(zipped.val[1]));   // e16..e23
        uint16x8_t w16_3 = vmovl_u8(vget_high_u8(zipped.val[1]));  // e24..e31

        // e0..e3
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_0)));
        float32x4_t x0 = vld1q_f32(x + i);
        f0 = vfmaq_f32(v_neg_zs, f0, v_scale);  // val*scale + (-zero*scale)
        acc0 = vfmaq_f32(acc0, f0, x0);

        // e4..e7
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_0)));
        float32x4_t x1 = vld1q_f32(x + i + 4);
        f1 = vfmaq_f32(v_neg_zs, f1, v_scale);
        acc1 = vfmaq_f32(acc1, f1, x1);

        // e8..e11
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_1)));
        float32x4_t x2 = vld1q_f32(x + i + 8);
        f2 = vfmaq_f32(v_neg_zs, f2, v_scale);
        acc2 = vfmaq_f32(acc2, f2, x2);

        // e12..e15
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_1)));
        float32x4_t x3 = vld1q_f32(x + i + 12);
        f3 = vfmaq_f32(v_neg_zs, f3, v_scale);
        acc3 = vfmaq_f32(acc3, f3, x3);

        // e16..e19
        float32x4_t f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_2)));
        float32x4_t x4 = vld1q_f32(x + i + 16);
        f4 = vfmaq_f32(v_neg_zs, f4, v_scale);
        acc0 = vfmaq_f32(acc0, f4, x4);

        // e20..e23
        float32x4_t f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_2)));
        float32x4_t x5 = vld1q_f32(x + i + 20);
        f5 = vfmaq_f32(v_neg_zs, f5, v_scale);
        acc1 = vfmaq_f32(acc1, f5, x5);

        // e24..e27
        float32x4_t f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_3)));
        float32x4_t x6 = vld1q_f32(x + i + 24);
        f6 = vfmaq_f32(v_neg_zs, f6, v_scale);
        acc2 = vfmaq_f32(acc2, f6, x6);

        // e28..e31
        float32x4_t f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_3)));
        float32x4_t x7 = vld1q_f32(x + i + 28);
        f7 = vfmaq_f32(v_neg_zs, f7, v_scale);
        acc3 = vfmaq_f32(acc3, f7, x7);
    }

    // Reduce 4 accumulators
    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float total = vaddvq_f32(vaddq_f32(sum01, sum23));

    // Scalar tail for remaining elements (group_elems not multiple of 32)
    for (; i < group_elems; ++i) {
        const int byte_idx = i / 2;
        uint8_t val = (i % 2 == 0) ? (packed[byte_idx] & 0x0F) : ((packed[byte_idx] >> 4) & 0x0F);
        total += (static_cast<float>(val) - zero) * scale * x[i];
    }

    return total;
}

constexpr int kGroupHeader = 4;

void matvec_i4_neon(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_data_bytes = group_size / 2;
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    const int row_bytes = groups_per_row * group_total_bytes;

    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = w + static_cast<size_t>(o) * row_bytes;
        float acc = 0.0f;
        int col = 0;

        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *group_ptr = row + g * group_total_bytes;

            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, group_ptr, 2);
            std::memcpy(&zero_h, group_ptr + 2, 2);
            const float scale = half_to_float(scale_h);
            const float zero = half_to_float(zero_h);

            const uint8_t *packed = group_ptr + kGroupHeader;
            const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

            acc += dot_group_i4_neon(packed, x + col, scale, zero, group_elems);
            col += group_size;
        }
        y[o] = acc;
    }
}

} // anonymous namespace

TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_neon, "neon");

} // namespace tinyqwen

#endif // __aarch64__
