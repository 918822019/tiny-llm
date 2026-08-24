// ============================================================================
// matvec_vq2_ref.cpp — VQ2（2-bit 块向量量化）weight-only matvec 参考实现
// ============================================================================
// 本文件是 2-bit 块向量量化权重矩阵乘向量的"标准答案"。
//
// 内存布局（与 runtime/tiny_format.h 的 kVQ2* 常量一致）：
//   每个权重张量 = [ 码本: [K=256, d=4] fp16 = 2048B ][ 索引区 ]
//   索引区：每行 in_dim/d 个 uint8；每个索引编码该行 d(=4) 个连续权重。
//   反量化：block b 的 d 个权重 = codebook[index_b]（一个 d 维 fp16 向量）。
//
// 块向量量化：码率 = log2(K)/d = log2(256)/4 = 2 bit/权重。
// 这正是 2.0bit 的"部署洁净点"：每 d=4 个权重恰 1 字节、字节对齐、免 bit-pack，
// 反量化只有一次码本行查表，没有位操作、没有乘加。
//
// 数值约定：double 累加（最保守路径），所有 VQ2 优化变体必须与本实现对齐后
// 才能声称"算对了"。永不修改、永不删除。
//
// 前置约束：in_dim 必须能被 kBlockDim(=4) 整除（导出器/加载器已保证）。
//
// 注册名："ref"——dispatch 中 matvec_vq2 的默认/兜底实现。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VQ2_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float

#include <cstdint>    // uint8_t, uint16_t
#include <cstring>    // std::memcpy（安全读取 fp16 码本）

namespace tinyqwen {

namespace {
constexpr int kBlockDim = 4;                                 // 块大小 d
constexpr int kCodebookEntries = 256;                         // K
constexpr int kCodebookBytes = kCodebookEntries * kBlockDim * 2; // [K,d] fp16 = 2048B
} // namespace

// ========================================================================
// matvec_vq2_ref() — VQ2 块向量量化权重矩阵乘向量参考实现
// ========================================================================
// 功能：计算 y[out_dim] = dequant(W_vq2)[out_dim, in_dim] @ x[in_dim]
// 参数：
//   w       — 指向 [码本 2048B][索引区] 的字节指针
//   x       — fp32 输入向量（长度 in_dim）
//   y       — fp32 输出向量（长度 out_dim，调用方分配）
//   out_dim — 输出维度（W 的行数）
//   in_dim  — 输入维度（W 的列数 = x 的长度，须被 4 整除）
// 算法步骤：
//   1. 读码本：[K, d] fp16 → fp32，加载一次，供所有行复用；
//   2. 外层遍历 out_dim 行；
//   3. 内层逐块：查表得 d 维码本向量 → 逐元素 double 乘累加。
// 返回值：无（结果写入 y）
void matvec_vq2_ref(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim) {
    // 1. 码本 [K, d] fp16 → fp32（一次性，跨行复用，驻留栈/L1）
    float cb[kCodebookEntries][kBlockDim];
    for (int k = 0; k < kCodebookEntries; ++k) {
        for (int j = 0; j < kBlockDim; ++j) {
            uint16_t h;
            std::memcpy(&h, w + (k * kBlockDim + j) * 2, 2); // 未对齐安全读
            cb[k][j] = half_to_float(h);
        }
    }

    // 2. 索引区紧跟码本之后；每行 n_blocks 个 uint8
    const int n_blocks = in_dim / kBlockDim;
    const uint8_t *idx = w + kCodebookBytes;

    // 3. 外层逐行：逐块查表 + double 累加
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        double acc = 0.0;
        for (int b = 0; b < n_blocks; ++b) {
            const float *cv = cb[row[b]];          // 查表得 d 维码本向量
            const int base = b * kBlockDim;
            for (int j = 0; j < kBlockDim; ++j) {
                acc += static_cast<double>(cv[j]) * static_cast<double>(x[base + j]);
            }
        }
        y[o] = static_cast<float>(acc);
    }
}

// 自注册进 dispatch：matvec_vq2 的 "ref" 实现（默认/兜底）
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_ref, "ref");

} // namespace tinyqwen
