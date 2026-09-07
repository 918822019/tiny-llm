// ============================================================================
// matvec_gptq_ref.cpp — 原生 GPTQ-INT4 matvec 参考实现：y = W @ x
// ============================================================================
// 本文件是 GPTQ 量化权重矩阵乘向量的"标准答案"（与 matvec_i4_ref 平行，
// 但走 AutoGPTQ 原生列主序打包，不重打包成 HQQ interleaved 布局）。
//
// in-band 块布局（见 tiny_format.h 的 GptqBlockOffsets）：
//   [ header: {u32 magic, u32 flags} ]                  flags bit0 = has_g_idx
//   [ scales:  [n_groups, out_dim] fp16 行主序 ]
//   [ qzeros:  [n_groups, out_dim] fp16 行主序 ]        （存 fp16，省 int4 解包）
//   [ g_idx:   [in_dim] u32 ]                           （has_g_idx 时存在）
//   [ qweight: [(in_dim/8), out_dim] u32 列主序 ]       AutoGPTQ：每个 int32
//                                                       装 8 个 uint4，对应同一
//                                                       列(out_dim)的 8 个连续
//                                                       in_dim 行；低 nibble=行0
// 反量化语义（与 AutoGPTQ/HF 一致）：
//   g = has_g_idx ? g_idx[col] : col / group_size
//   nibble(row,col) = (qweight[(col/8)*out_dim + row] >> ((col%8)*4)) & 0xF
//   val(row,col) = (float(nibble) - qzero[g*out_dim+row]) * scale[g*out_dim+row]
//
// 数值约定：double 累加（与 matvec_i4_ref 同），所有 GPTQ 优化变体必须与本
// 实现对齐后才能声称"算对了"。永不修改、永不删除。
//
// 注册名："ref"——dispatch 中 matvec_gptq 的默认/兜底实现。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_GPTQ_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float

#include <cstring>    // std::memcpy（安全读取未对齐的 fp16/u32）

namespace tinyqwen {

namespace {
// GPTQ in-band 布局常量（与 runtime/tiny_format.h 的 GptqBlockOffsets 契约一致；
// kernels 层不反向 include runtime 头，故在此镜像定义——与 matvec_i4_ref
// 本地定义 kGroupHeader=4 同款做法）。
constexpr int kGptqHeaderBytes = 8;  // {u32 magic, u32 flags}
constexpr int kGptqPackInts = 8;    // 每个 int32 装 8 个 uint4

// 安全读取未对齐的 fp16 / u32（in-band 子段起点未必对齐）
inline float read_fp16(const uint8_t *p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return half_to_float(h);
}
inline uint32_t read_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
} // namespace

// ========================================================================
// matvec_gptq_ref() — GPTQ-INT4 权重矩阵乘向量参考实现
// ========================================================================
// 功能：计算 y[out_dim] = W_gptq[out_dim, in_dim] @ x[in_dim]
// 参数：
//   w          — GPTQ in-band 块起点（header 起始）
//   x          — fp32 输入向量（长度 in_dim）
//   y          — fp32 输出向量（长度 out_dim，调用方分配）
//   out_dim    — 输出维度（W 的行数 = 专家投影的 inter）
//   in_dim     — 输入维度（W 的列数 = hidden）
//   group_size — GPTQ 分组大小（要求 in_dim % group_size == 0 且 in_dim % 8 == 0）
void matvec_gptq_ref(const uint8_t *w, const float *x, float *y,
                     int out_dim, int in_dim, int group_size) {
    // ---- 解析 in-band 块各段偏移（与 tiny_format.h 约定一致）----
    const uint32_t flags = read_u32(w + 4);
    const bool has_g_idx = (flags & 1u) != 0;
    const int n_groups = in_dim / group_size;
    const size_t sq = static_cast<size_t>(n_groups) * out_dim * 2; // scales/qzeros 各占
    const uint8_t *scales = w + kGptqHeaderBytes;
    const uint8_t *qzeros = scales + sq;
    const uint8_t *g_idx = has_g_idx ? (qzeros + sq) : nullptr;
    const uint8_t *qweight = has_g_idx ? (g_idx + static_cast<size_t>(in_dim) * 4)
                                      : (qzeros + sq);
    const int pack = kGptqPackInts; // 8 个 uint4 装一个 int32

    // ---- 逐输出行累加 ----
    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        // 按 int32 字（8 个连续 in_dim 行）推进；同一字对应同一列 o
        for (int c8 = 0; c8 < in_dim / pack; ++c8) {
            const uint32_t word = read_u32(qweight +
                (static_cast<size_t>(c8) * out_dim + o) * 4);
            for (int k = 0; k < pack; ++k) {
                const int c = c8 * pack + k;
                const uint32_t nibble = (word >> (k * 4)) & 0xFu;
                const int g = has_g_idx
                    ? static_cast<int>(read_u32(g_idx + static_cast<size_t>(c) * 4))
                    : c / group_size;
                const float s = read_fp16(scales + (static_cast<size_t>(g) * out_dim + o) * 2);
                const float z = read_fp16(qzeros + (static_cast<size_t>(g) * out_dim + o) * 2);
                const float dequant = (static_cast<float>(nibble) - z) * s;
                acc += static_cast<double>(dequant) * static_cast<double>(x[c]);
            }
        }
        y[o] = static_cast<float>(acc);
    }
}

// 自注册进 dispatch：matvec_gptq 的 "ref" 实现（默认/兜底）
TINYQWEN_MATVEC_GPTQ_VARIANT(matvec_gptq_ref, "ref");

} // namespace tinyqwen
