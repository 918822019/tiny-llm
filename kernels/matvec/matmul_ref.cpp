// Matmul reference: Y[M,N] = W[M,K] × X[K,N]
//
// W row-major [M,K], X column-major [K,N] (each col = one token), Y column-major [M,N].
// 本质是对 N 列各做一次 matvec，但写成独立 kernel 便于后续 tiling 优化。

#include "dispatch.h"
#include "ref_ops.h"

#include <cstring>

namespace tinyqwen {
namespace {

void matmul_ref(const float *w, const float *x, float *y, int M, int K, int N) {
    for (int col = 0; col < N; ++col) {
        const float *xc = x + static_cast<size_t>(col) * K;
        float *yc = y + static_cast<size_t>(col) * M;
        for (int row = 0; row < M; ++row) {
            double acc = 0.0;
            const float *wr = w + static_cast<size_t>(row) * K;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(wr[k]) * static_cast<double>(xc[k]);
            }
            yc[row] = static_cast<float>(acc);
        }
    }
}

} // namespace
TINYQWEN_MATMUL_VARIANT(matmul_ref, "ref");
} // namespace tinyqwen
