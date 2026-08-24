// ============================================================================
// matvec_vq2_neon.cpp — VQ2（2-bit 块向量量化）matvec：y = W @ x —— NEON 版
// ============================================================================
// matvec_vq2_ref 的 NEON 优化变体。布局与反量化完全一致：
//   [ 码本 [K=256, d=4] fp16 = 2048B ][ 索引: out_dim × (in_dim/4) × uint8 ]
//   block b 的 d=4 个权重 = codebook[index_b]（一个 4 维向量）。
//
// 块大小 d=4 恰好 = 一个 float32x4，因此本变体是真正的 4 宽 SIMD：
//   每块：标量查表得到码本行指针（4 个 fp32，栈上已展开）→ vld1q_f32 载入
//   → vld1q_f32 载入对应 4 个激活 → vfmaq_f32 累加。
//   查表仍是标量（NEON 无 256 项表查询指令），但每次查表喂 4 路 FMA，
//   摊薄了查表开销；码本 4KB(fp32) 常驻 L1。
//
// 权重流量 = 1 字节/块 = 2 bit/权重，贴近内存带宽墙；算力结构极简（查表 + FMA），
// 无位 unpack——这正是 2.0bit"部署洁净点"的内核形态。
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
} // namespace

// ========================================================================
// matvec_vq2_neon() — VQ2 matvec NEON 版（4 宽 FMA，fp32 累加）
// ========================================================================
// 功能：y[out_dim] = dequant(W_vq2) @ x[in_dim]，与 ref 数值等价（累加序不同，
//       存在可忽略的 fp 舍入差；测试用容差对齐，不要求逐位一致）。
void matvec_vq2_neon(const uint8_t *w, const float *x, float *y,
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

    // 3. 逐行：逐块查表 + 4 宽 FMA
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float *cv = cb[row[b]];                 // 标量查表 → 码本行指针
            const float32x4_t cv4 = vld1q_f32(cv);        // 4 个码本值
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim); // 4 个激活
            acc = vfmaq_f32(acc, cv4, x4);                // 4 宽 FMA 累加
        }
        y[o] = vaddvq_f32(acc); // 横向求和
    }
}

// 自注册进 dispatch：matvec_vq2 的 "neon" 实现（快速变体）
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon, "neon");

} // namespace tinyqwen

#endif // __aarch64__
