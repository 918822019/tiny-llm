// FP16 weight-only GEMM reference implementation.
// Y[M,N] = W[M,K] * X[K,N], with W row-major and X/Y token-column-major.

#include "dispatch.h"
#include "ref_ops.h"

#include <cstddef>
#include <cstdint>

namespace tinyqwen {

void matmul_f16_ref(const uint16_t *w, const float *x, float *y,
                    int M, int K, int N) {
    for (int c = 0; c < N; ++c) {
        const float *xc = x + static_cast<size_t>(c) * K;
        float *yc = y + static_cast<size_t>(c) * M;
        for (int row = 0; row < M; ++row) {
            const uint16_t *wr = w + static_cast<size_t>(row) * K;
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<double>(half_to_float(wr[k])) * xc[k];
            yc[row] = static_cast<float>(acc);
        }
    }
}

TINYQWEN_MATMUL_F16_VARIANT(matmul_f16_ref, "ref");

} // namespace tinyqwen
