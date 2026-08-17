// INT4 weight-only matvec reference：y = W @ x
//
// 非对称 uint4 [0,15]，per-group scale + zero_point（均 fp16），interleaved 布局。
// 每组 68 字节：[scale_fp16(2B) | zero_fp16(2B) | packed_uint4(group_size/2 B)]。
// 反量化：float_val = (uint4_val - zero) * scale。
// 低 nibble 在前：byte & 0x0F = 偶数下标，byte >> 4 = 奇数下标。
//
// 正确性锚：double 累加，所有优化变体必须与本实现对齐后才能声称"算对了"。

#include "dispatch.h"
#include "ref_ops.h"

#include <cstring>

namespace tinyqwen {

namespace {
constexpr int kGroupHeader = 4; // scale(fp16, 2B) + zero(fp16, 2B)
} // namespace

void matvec_i4_ref(const uint8_t *w, const float *x, float *y,
                   int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_data_bytes = group_size / 2;
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    const int row_bytes = groups_per_row * group_total_bytes;

    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = w + static_cast<size_t>(o) * row_bytes;
        double acc = 0.0;
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

            for (int i = 0; i < group_elems; ++i) {
                const int byte_idx = i / 2;
                uint8_t val;
                if (i % 2 == 0) {
                    val = packed[byte_idx] & 0x0F;
                } else {
                    val = (packed[byte_idx] >> 4) & 0x0F;
                }
                const float dequant = (static_cast<float>(val) - zero) * scale;
                acc += static_cast<double>(dequant) * static_cast<double>(x[col + i]);
            }
            col += group_size;
        }
        y[o] = static_cast<float>(acc);
    }
}

TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_ref, "ref");

} // namespace tinyqwen
