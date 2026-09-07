// ============================================================================
// matmul_gptq_neon.cpp — GPTQ-INT4 批量 GEMM NEON 实现
// ============================================================================
// 结构与 matvec_gptq_neon 相同（向量化在 **o** 方向），只是外面多一层列循环。
//
// 为什么向量化在 o 而不是列：X 是列主序 [K, N]，元素 (row r, col c) 在
// `c*K + r`。固定 r、变化 c 的步长是 K —— **不连续**，无法向量化。
// 而 qweight 是 [(K/8), M]，固定 c8、变化 o **连续**，可向量化。
// （初版曾误以为可对列向量化，实测结果全错。）
//
// 复用 matvec_gptq_neon 的三处优化：
//   ① c8-outer / o-inner 遍历（连续内存）
//   ② o 方向分块 64，累加器留 L1
//   ③ 因式分解：Σ_k((nib_k - z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]
//      其中 Σ_k(x_k) 只依赖 (c8, col)，每列每 c8 算一次标量。
//
// 批量的收益来源：专家权重加载一次后，对该专家的所有 token 复用（外层由
// 调用方按专家分组），本 kernel 只负责多列 GEMM。
//
// 布局与数值语义同 matmul_gptq_ref（double 累加），须与之对齐。
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include "ref_ops.h"

#include <arm_neon.h>
#include <cstring>
#include <vector>

namespace tinyqwen {

namespace {
constexpr int kGptqHeaderBytes = 8;
constexpr int kGptqPackInts = 8;
constexpr int kOBlock = 64;    // o 方向分块，累加器留 L1

inline uint32_t gread_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
} // namespace

void matmul_gptq_neon(const uint8_t *w, const float *x, float *y,
                      int M, int K, int N, int group_size) {
    const uint32_t flags = gread_u32(w + 4);
    const bool has_g_idx = (flags & 1u) != 0;
    const int ng = K / group_size;
    const size_t sq = static_cast<size_t>(ng) * M * 2;
    const uint8_t *scales = w + kGptqHeaderBytes;
    const uint8_t *qzeros = scales + sq;
    const uint8_t *g_idx = has_g_idx ? (qzeros + sq) : nullptr;
    const uint8_t *qweight = has_g_idx ? (g_idx + static_cast<size_t>(K) * 4)
                                       : (qzeros + sq);
    const int n_c8 = K / kGptqPackInts;

    std::vector<int> g_of_c8(static_cast<size_t>(n_c8));
    for (int c8 = 0; c8 < n_c8; ++c8) {
        g_of_c8[static_cast<size_t>(c8)] =
            g_idx ? static_cast<int>(gread_u32(g_idx + static_cast<size_t>(c8 * 8) * 4))
                  : (c8 * kGptqPackInts) / group_size;
    }

    const uint32x4_t mask = vdupq_n_u32(0xFu);

    // 逐列（每列一个 token）
    for (int col = 0; col < N; ++col) {
        const float *xc = x + static_cast<size_t>(col) * K;
        float *yc = y + static_cast<size_t>(col) * M;

        // sx8[c8] = Σ_{k=0..7} xc[c8*8 + k]，只依赖 (c8, col)
        std::vector<float> sx8(static_cast<size_t>(n_c8));
        for (int c8 = 0; c8 < n_c8; ++c8) {
            const float *p = xc + c8 * kGptqPackInts;
            sx8[static_cast<size_t>(c8)] =
                p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }

        // o 方向分块
        for (int ob = 0; ob < M; ob += kOBlock) {
            const int blen = (M - ob < kOBlock) ? M - ob : kOBlock;
            float32x4_t acc[16];
            for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0.0f);
            const int n4 = blen & ~3;

            for (int c8 = 0; c8 < n_c8; ++c8) {
                const int g = g_of_c8[static_cast<size_t>(c8)];
                const uint32_t *wrow =
                    reinterpret_cast<const uint32_t *>(qweight) +
                    static_cast<size_t>(c8) * M + ob;
                const float16_t *srow =
                    reinterpret_cast<const float16_t *>(scales) +
                    static_cast<size_t>(g) * M + ob;
                const float16_t *zrow =
                    reinterpret_cast<const float16_t *>(qzeros) +
                    static_cast<size_t>(g) * M + ob;
                const float sx = sx8[static_cast<size_t>(c8)];
                const float *xk = xc + c8 * kGptqPackInts;

                const float32x4_t xk0 = vdupq_n_f32(xk[0]);
                const float32x4_t xk1 = vdupq_n_f32(xk[1]);
                const float32x4_t xk2 = vdupq_n_f32(xk[2]);
                const float32x4_t xk3 = vdupq_n_f32(xk[3]);
                const float32x4_t xk4 = vdupq_n_f32(xk[4]);
                const float32x4_t xk5 = vdupq_n_f32(xk[5]);
                const float32x4_t xk6 = vdupq_n_f32(xk[6]);
                const float32x4_t xk7 = vdupq_n_f32(xk[7]);

                for (int o4 = 0; o4 < n4; o4 += 4) {
                    const uint32x4_t w4 = vld1q_u32(wrow + o4);
                    float32x4_t dot4 =
                        vfmaq_f32(vdupq_n_f32(0.0f),
                                  vcvtq_f32_u32(vandq_u32(w4, mask)), xk0);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 4), mask)), xk1);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 8), mask)), xk2);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 12), mask)), xk3);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 16), mask)), xk4);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 20), mask)), xk5);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 24), mask)), xk6);
                    dot4 = vfmaq_f32(dot4, vcvtq_f32_u32(vshrq_n_u32(w4, 28)), xk7);

                    const float32x4_t s4 = vcvt_f32_f16(vld1_f16(srow + o4));
                    const float32x4_t z4 = vcvt_f32_f16(vld1_f16(zrow + o4));
                    const float32x4_t term = vsubq_f32(dot4, vmulq_n_f32(z4, sx));
                    acc[o4 >> 2] = vfmaq_f32(acc[o4 >> 2], s4, term);
                }
            }

            for (int o4 = 0; o4 < n4; o4 += 4) vst1q_f32(yc + ob + o4, acc[o4 >> 2]);
            // 标量尾部（blen % 4）
            for (int o = n4; o < blen; ++o) {
                double a = 0.0;
                for (int c8 = 0; c8 < n_c8; ++c8) {
                    const int g = g_of_c8[static_cast<size_t>(c8)];
                    const uint32_t word = gread_u32(
                        qweight + (static_cast<size_t>(c8) * M + ob + o) * 4);
                    uint16_t sh, zh;
                    std::memcpy(&sh, scales + (static_cast<size_t>(g) * M + ob + o) * 2, 2);
                    std::memcpy(&zh, qzeros + (static_cast<size_t>(g) * M + ob + o) * 2, 2);
                    const float s = half_to_float(sh);
                    const float z = half_to_float(zh);
                    const float *xk = xc + c8 * kGptqPackInts;
                    double dot = 0.0;
                    for (int k = 0; k < kGptqPackInts; ++k)
                        dot += static_cast<double>((word >> (k * 4)) & 0xFu) *
                               static_cast<double>(xk[k]);
                    a += static_cast<double>(s) *
                         (dot - static_cast<double>(z) *
                                static_cast<double>(sx8[static_cast<size_t>(c8)]));
                }
                yc[ob + o] = static_cast<float>(a);
            }
        }
    }
}

TINYQWEN_MATMUL_GPTQ_VARIANT(matmul_gptq_neon, "neon");

} // namespace tinyqwen

#else // 非 ARM：兜底 ref

namespace tinyqwen {
void matmul_gptq_neon(const uint8_t *w, const float *x, float *y,
                      int M, int K, int N, int group_size) {
    matmul_gptq_ref(w, x, y, M, K, N, group_size);
}
TINYQWEN_MATMUL_GPTQ_VARIANT(matmul_gptq_neon, "neon");
} // namespace tinyqwen

#endif
