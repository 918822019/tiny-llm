// ============================================================================
// matmul_gptq_ref.cpp — GPTQ-INT4 批量 GEMM 参考实现：Y[M,N] = W[M,K] × X[K,N]
// ============================================================================
// MoE 批量 prefill 的基础：同一专家的权重被多个 token 共享，一次加载 + 一次
// GEMM 处理全部 token，而不是逐 token 各读一次盘、各算一次 matvec。
//
// 布局与反量化语义同 matvec_gptq_ref（in-band 块）：
//   [header 8B {magic, flags}][scales fp16 (ng,out)][qzeros fp16 (ng,out)]
//   [g_idx u32 (in) 可选][qweight u32 (in/8,out) 列主序]
//
// 沿用因式分解：同一 u32 字的 8 个 nibble 共享 s/z，故
//   Σ_k ((nib_k - z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]
// 其中 Σ_k(x_k) 只依赖 (c8, col)，可预算。
//
// X / Y 列主序：第 col 列 = 一个 token 的向量（与 matmul_i4_ref 同约定）。
// 数值约定：double 累加（与 matvec_gptq_ref 一致），所有优化变体须与之对齐。
// ============================================================================

#include "dispatch.h"
#include "ref_ops.h"

#include <cstring>
#include <vector>

namespace tinyqwen {

namespace {
constexpr int kGptqHeaderBytes = 8;
constexpr int kGptqPackInts = 8;

inline uint32_t read_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline float read_fp16(const uint8_t *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return half_to_float(h);
}
} // namespace

void matmul_gptq_ref(const uint8_t *w, const float *x, float *y,
                     int M, int K, int N, int group_size) {
    const uint32_t flags = read_u32(w + 4);
    const bool has_g_idx = (flags & 1u) != 0;
    const int ng = K / group_size;
    const size_t sq = static_cast<size_t>(ng) * M * 2;
    const uint8_t *scales = w + kGptqHeaderBytes;
    const uint8_t *qzeros = scales + sq;
    const uint8_t *g_idx = has_g_idx ? (qzeros + sq) : nullptr;
    const uint8_t *qweight = has_g_idx ? (g_idx + static_cast<size_t>(K) * 4)
                                       : (qzeros + sq);

    const int n_c8 = K / kGptqPackInts;

    // 预算 sx8[c8][col] = Σ_{k=0..7} x[col*K + c8*8 + k]
    // 只依赖 (c8, col)，与 M 无关 —— 这是因式分解能省掉 8 倍 s/z 运算的前提。
    std::vector<float> sx8(static_cast<size_t>(n_c8) * N);
    for (int col = 0; col < N; ++col) {
        const float *xc = x + static_cast<size_t>(col) * K;
        for (int c8 = 0; c8 < n_c8; ++c8) {
            const float *p = xc + c8 * kGptqPackInts;
            sx8[static_cast<size_t>(c8) * N + col] =
                p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }
    }

    // 每个 c8 对应的组号（contiguous 时 = c8*8/group_size）
    std::vector<int> g_of_c8(static_cast<size_t>(n_c8));
    for (int c8 = 0; c8 < n_c8; ++c8) {
        g_of_c8[static_cast<size_t>(c8)] =
            g_idx ? static_cast<int>(read_u32(g_idx + static_cast<size_t>(c8 * 8) * 4))
                  : (c8 * kGptqPackInts) / group_size;
    }

    // 逐 token（列）计算：Y 的第 col 列
    for (int col = 0; col < N; ++col) {
        const float *xc = x + static_cast<size_t>(col) * K;
        float *yc = y + static_cast<size_t>(col) * M;
        for (int o = 0; o < M; ++o) {
            double acc = 0.0;
            for (int c8 = 0; c8 < n_c8; ++c8) {
                const int g = g_of_c8[static_cast<size_t>(c8)];
                const uint32_t word = read_u32(
                    qweight + (static_cast<size_t>(c8) * M + o) * 4);
                const float s = read_fp16(scales + (static_cast<size_t>(g) * M + o) * 2);
                const float z = read_fp16(qzeros + (static_cast<size_t>(g) * M + o) * 2);
                const float *p = xc + c8 * kGptqPackInts;
                double dot = 0.0;
                for (int k = 0; k < kGptqPackInts; ++k) {
                    const uint32_t nib = (word >> (k * 4)) & 0xFu;
                    dot += static_cast<double>(nib) * static_cast<double>(p[k]);
                }
                const double sx = static_cast<double>(
                    sx8[static_cast<size_t>(c8) * N + col]);
                acc += static_cast<double>(s) *
                       (dot - static_cast<double>(z) * sx);
            }
            yc[o] = static_cast<float>(acc);
        }
    }
}

TINYQWEN_MATMUL_GPTQ_VARIANT(matmul_gptq_ref, "ref");

} // namespace tinyqwen
