// Matmul I4 reference: Y[M,N] = W_i4[M,K] × X[K,N]
//
// W interleaved INT4 packed (same layout as matvec_i4_ref), X col-major [K,N], Y col-major [M,N].

#include "dispatch.h"
#include "ref_ops.h"

#include <cstring>

namespace tinyqwen {
namespace {

constexpr int kGroupHeader = 4;

void matmul_i4_ref(const uint8_t *w, const float *x, float *y,
                   int M, int K, int N, int group_size) {
    const int groups_per_row = (K + group_size - 1) / group_size;
    const int group_data_bytes = group_size / 2;
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    const int row_bytes = groups_per_row * group_total_bytes;

    for (int col = 0; col < N; ++col) {
        const float *xc = x + static_cast<size_t>(col) * K;
        float *yc = y + static_cast<size_t>(col) * M;

        for (int row = 0; row < M; ++row) {
            const uint8_t *rp = w + static_cast<size_t>(row) * row_bytes;
            double acc = 0.0;
            int k = 0;

            for (int g = 0; g < groups_per_row; ++g) {
                const uint8_t *gp = rp + g * group_total_bytes;
                uint16_t scale_h, zero_h;
                std::memcpy(&scale_h, gp, 2);
                std::memcpy(&zero_h, gp + 2, 2);
                const float scale = half_to_float(scale_h);
                const float zero = half_to_float(zero_h);

                const uint8_t *packed = gp + kGroupHeader;
                const int elems = (k + group_size <= K) ? group_size : (K - k);

                for (int i = 0; i < elems; ++i) {
                    uint8_t val = (i % 2 == 0) ? (packed[i / 2] & 0x0F)
                                               : ((packed[i / 2] >> 4) & 0x0F);
                    float dequant = (static_cast<float>(val) - zero) * scale;
                    acc += static_cast<double>(dequant) * static_cast<double>(xc[k + i]);
                }
                k += group_size;
            }
            yc[row] = static_cast<float>(acc);
        }
    }
}

} // namespace
TINYQWEN_MATMUL_I4_VARIANT(matmul_i4_ref, "ref");
} // namespace tinyqwen
