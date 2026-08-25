// ============================================================================
// matvec_vq2_neon_mr.cpp — VQ2 matvec：neon + 多行 ILP（multi-row）
// ============================================================================
// matvec_vq2_neon 的归因阶梯下一层。布局与反量化完全一致：
//   [ 码本 [K=256, d=4] fp16 = 2048B ][ 索引: out_dim × (in_dim/4) × uint8 ]
//
// 上一层（neon）的瓶颈：单行单累加器——
//   acc = vfma(acc, cb[row[b]], x[b])
// 每个 FMA 依赖上一个的结果，串行依赖链被 FMA 延迟（~3-4 周期）卡住；
// 且每行各自重复加载同一份 x4。
//
// 本层只加一个技术：**4 行并行**（对齐 i4 族 sdot2 的"2-row 并行"路径）：
//   - 4 个独立累加器 a0..a3 → 4 条互不依赖的 FMA 链，隐藏延迟；
//   - 每块只加载一次 x4，4 行共享 → x 侧加载端口压力降为 1/4；
//   - 4 路索引查表相互独立，可与 FMA 流水重叠。
// 码本 4KB 仍常驻栈/L1；行数不足 4 的尾部退回单行路径（数值同款）。
//
// 仅 aarch64 构建注册；其他平台编译为空翻译单元。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VQ2_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h> // NEON intrinsic
#include <cstdint>    // uint8_t, uint16_t
#include <cstring>    // std::memcpy

namespace tinyqwen {

namespace {
constexpr int kBlockDim = 4;
constexpr int kCodebookEntries = 256;
constexpr int kCodebookBytes = kCodebookEntries * kBlockDim * 2; // 2048B
constexpr int kRows = 4;                                          // 多行并行度
} // namespace

// ========================================================================
// matvec_vq2_neon_mr() — VQ2 matvec：4 行并行 FMA 链，共享 x4 加载
// ========================================================================
// 数值语义与 matvec_vq2_neon 完全一致（同一查表反量化 + fp32 累加，
// 仅累加序不同），测试用容差对齐。
void matvec_vq2_neon_mr(const uint8_t *w, const float *x, float *y,
                        int out_dim, int in_dim) {
    // 1. 码本 [K, d] fp16 → fp32（一次性，4KB 驻留栈/L1，跨行复用）
    float cb[kCodebookEntries][kBlockDim];
    for (int k = 0; k < kCodebookEntries; ++k) {
        for (int j = 0; j < kBlockDim; ++j) {
            uint16_t h;
            std::memcpy(&h, w + (k * kBlockDim + j) * 2, 2);
            cb[k][j] = half_to_float(h);
        }
    }

    // 2. 索引区紧跟码本之后；每行 n_blocks 个 uint8
    const int n_blocks = in_dim / kBlockDim;
    const uint8_t *idx = w + kCodebookBytes;

    // 3. 主体：每次 4 行 —— 4 条独立 FMA 链 + 共享 x4 加载
    int o = 0;
    for (; o + kRows <= out_dim; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim); // 4 行共享
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), x4);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), x4);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), x4);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), x4);
        }

        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }

    // 4. 尾部（不足 4 行）：单行路径，与 neon 内循环同款
    for (; o < out_dim; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[row[b]]), x4);
        }
        y[o] = vaddvq_f32(acc);
    }
}

// 自注册进 dispatch：matvec_vq2 的 "neon_mr" 实现
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon_mr, "neon_mr");

} // namespace tinyqwen

#endif // __aarch64__
