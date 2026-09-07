#pragma once

// ============================================================================
// matvec_gptq_neon_common.h — GPTQ NEON 内核的共享内层逻辑
// ============================================================================
// neon 单线程与 neon_mt 多线程两个变体共用同一份"块处理"实现，只有调度方式
// 不同（单线程顺序遍历 o_block vs 线程池动态领取 o_block）。抽成共享头避免
// 两份 100 行内层循环各自演化后数值行为分叉——与 kernels/ref_ops.h、
// kernels/gdn_ops.h 同款做法。
//
// 布局契约与反量化语义见 matvec_gptq_ref.cpp（正确性 oracle，永不修改）。
// ============================================================================

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cstdint>
#include <cstring>

namespace tinyqwen {

// 前置声明：ref 定义在 matvec_gptq_ref.cpp 的 namespace tinyqwen 内（非匿名
// 空间），有外部链接。必须声明在匿名空间**之外**——放进匿名空间会给它内部
// 链接，链接期找不到真正的定义。kernels 层不互相 include 实现文件，故在此声明。
void matvec_gptq_ref(const uint8_t *w, const float *x, float *y, int out_dim,
                     int in_dim, int group_size);

namespace {

constexpr int kGptqHeaderBytes = 8;  // {u32 magic, u32 flags}
constexpr int kGptqPackInts = 8;     // 每个 int32 装 8 个 uint4

// o 方向分块大小。取 64 的理由：qweight 是 [in_dim/8, out_dim]，固定 c8 变化 o
// 才是连续内存；64 个 u32 = 256 B = 4 条完整 64 B cache line，既不浪费行也不
// 让累加器溢出寄存器（16 个 float32x4_t）。
constexpr int kGptqOBlock = 64;

inline uint32_t gptq_read_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// 解析 in-band 块各段起点。与 matvec_gptq_ref 的偏移算式逐项一致。
struct GptqBlockView {
    const uint8_t *scales = nullptr;
    const uint8_t *qzeros = nullptr;
    const uint8_t *g_idx = nullptr;   // has_g_idx 为假时是 nullptr
    const uint8_t *qweight = nullptr;
    int n_groups = 0;
    bool has_g_idx = false;
};

inline GptqBlockView parse_gptq_block(const uint8_t *w, int out_dim, int in_dim,
                                      int group_size) {
    GptqBlockView v;
    v.has_g_idx = (gptq_read_u32(w + 4) & 1u) != 0;
    v.n_groups = in_dim / group_size;
    const size_t sq = static_cast<size_t>(v.n_groups) * out_dim * 2;
    v.scales = w + kGptqHeaderBytes;
    v.qzeros = v.scales + sq;
    v.g_idx = v.has_g_idx ? (v.qzeros + sq) : nullptr;
    v.qweight = v.has_g_idx ? (v.g_idx + static_cast<size_t>(in_dim) * 4)
                            : (v.qzeros + sq);
    return v;
}

// g_idx 是否"每 8 行同组"。这是因式分解成立的前提：一个 u32 字装 8 个连续
// in_dim 行，只有这 8 行共享同一组时，scale/zero 才能提到 Σ 外面。
// act-order GPTQ 的 g_idx 是排列，可能打破这一点 —— 那种情况必须走慢路径，
// 否则算错且不报错（本仓库原则：绝不让坏数据悄悄流进后面的计算）。
inline bool gptq_g_idx_uniform(const uint8_t *g_idx, int in_dim) {
    if (g_idx == nullptr) return true;  // 无 g_idx 即 contiguous，天然均匀
    const int n_c8 = in_dim / kGptqPackInts;
    for (int c8 = 0; c8 < n_c8; ++c8) {
        const uint32_t g0 = gptq_read_u32(g_idx + static_cast<size_t>(c8 * 8) * 4);
        for (int k = 1; k < kGptqPackInts; ++k) {
            const uint32_t gk =
                gptq_read_u32(g_idx + static_cast<size_t>(c8 * 8 + k) * 4);
            if (gk != g0) return false;
        }
    }
    return true;
}

// ========================================================================
// gptq_neon_block() — 处理 [ob, ob+blen) 这一段输出列
// ========================================================================
// 因式分解（核心优化）：同一个 u32 字的 8 个 nibble 共享 s 与 z，故
//   Σ_k ((nib_k - z)·s·x_k) = s · [ Σ_k(nib_k·x_k) - z · Σ_k(x_k) ]
// 其中 Σ_k(x_k) 只依赖 c8、与 o 无关，已由调用方预算进 sx8。
// 于是每 8 个权重只需 1 次乘 + 1 次减 + 1 次 FMA，而不是 8 次减+乘+累加。
//
// 累加器为 fp32（ref 用 double），故与 ref 有 fp32 舍入级差异；容差由
// tests/test_gptq_matvec.cpp 实测确定。
inline void gptq_neon_block(const GptqBlockView &bv, const float *x, float *y,
                            const float *sx8, const int *g_of_c8, int out_dim,
                            int in_dim, int ob, int blen) {
    const int n_c8 = in_dim / kGptqPackInts;
    const uint8_t *qw = bv.qweight;
    const uint8_t *sc = bv.scales;
    const uint8_t *qz = bv.qzeros;

    // 16 个独立 fp32 累加器（blen<=64 → 至多 16 个 float32x4_t）
    float32x4_t acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = vdupq_n_f32(0.0f);

    const int n4 = blen & ~3;  // 4 的倍数部分走 SIMD，余数走标量尾部

    for (int c8 = 0; c8 < n_c8; ++c8) {
        const int g = g_of_c8[c8];
        const float sx = sx8[c8];
        const float *x8 = x + c8 * kGptqPackInts;
        // 固定 c8、变化 o → qweight / scales / qzeros 三者都是连续内存
        const uint32_t *wrow =
            reinterpret_cast<const uint32_t *>(qw +
                static_cast<size_t>(c8) * out_dim * 4) + ob;
        const float16_t *srow =
            reinterpret_cast<const float16_t *>(sc +
                static_cast<size_t>(g) * out_dim * 2) + ob;
        const float16_t *zrow =
            reinterpret_cast<const float16_t *>(qz +
                static_cast<size_t>(g) * out_dim * 2) + ob;

        // 8 个 x 值广播成向量：与 w4 的 8 次移位解包配对
        const float32x4_t xk0 = vdupq_n_f32(x8[0]);
        const float32x4_t xk1 = vdupq_n_f32(x8[1]);
        const float32x4_t xk2 = vdupq_n_f32(x8[2]);
        const float32x4_t xk3 = vdupq_n_f32(x8[3]);
        const float32x4_t xk4 = vdupq_n_f32(x8[4]);
        const float32x4_t xk5 = vdupq_n_f32(x8[5]);
        const float32x4_t xk6 = vdupq_n_f32(x8[6]);
        const float32x4_t xk7 = vdupq_n_f32(x8[7]);
        const uint32x4_t mask = vdupq_n_u32(0xFu);

        for (int o4 = 0; o4 < n4; o4 += 4) {
            const uint32x4_t w4 = vld1q_u32(wrow + o4);
            // Σ_k nib_k·x_k，4 列并行。k=0 无需移位（vshrq_n_u32 要求 1..32）
            float32x4_t dot4 =
                vfmaq_f32(vdupq_n_f32(0.0f),
                          vcvtq_f32_u32(vandq_u32(w4, mask)), xk0);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 4), mask)), xk1);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 8), mask)), xk2);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 12), mask)), xk3);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 16), mask)), xk4);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 20), mask)), xk5);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 24), mask)), xk6);
            dot4 = vfmaq_f32(dot4,
                             vcvtq_f32_u32(vandq_u32(vshrq_n_u32(w4, 28), mask)), xk7);

            const float32x4_t s4 = vcvt_f32_f16(vld1_f16(srow + o4));
            const float32x4_t z4 = vcvt_f32_f16(vld1_f16(zrow + o4));
            // s·(dot8 - z·sx8)
            const float32x4_t term = vsubq_f32(dot4, vmulq_n_f32(z4, sx));
            acc[o4 >> 2] = vfmaq_f32(acc[o4 >> 2], s4, term);
        }
    }

    // 写回 SIMD 部分
    for (int o4 = 0; o4 < n4; o4 += 4) {
        vst1q_f32(y + ob + o4, acc[o4 >> 2]);
    }
    // 标量尾部（blen % 4）：同样的因式分解，只是不向量化
    for (int o = n4; o < blen; ++o) {
        float a = 0.0f;
        for (int c8 = 0; c8 < n_c8; ++c8) {
            const int g = g_of_c8[c8];
            const uint32_t word = gptq_read_u32(
                qw + (static_cast<size_t>(c8) * out_dim + ob + o) * 4);
            const float *x8 = x + c8 * kGptqPackInts;
            float dot = 0.0f;
            for (int k = 0; k < kGptqPackInts; ++k) {
                const uint32_t nib = (word >> (k * 4)) & 0xFu;
                dot += static_cast<float>(nib) * x8[k];
            }
            uint16_t sh, zh;
            std::memcpy(&sh, sc + (static_cast<size_t>(g) * out_dim + ob + o) * 2, 2);
            std::memcpy(&zh, qz + (static_cast<size_t>(g) * out_dim + ob + o) * 2, 2);
            const float s = static_cast<float>(*reinterpret_cast<const float16_t *>(&sh));
            const float z = static_cast<float>(*reinterpret_cast<const float16_t *>(&zh));
            a += s * (dot - z * sx8[c8]);
        }
        y[ob + o] = a;
    }
}

// ========================================================================
// gptq_neon_slow_block() — act-order（g_idx 非均匀）兜底路径
// ========================================================================
// 无法因式分解：每个 nibble 的组号不同，s/z 必须逐 nibble 取。向量化仍用在
// "同一列 o 的 8 个 nibble × x" 上，但 s/z 逐元素读。
inline void gptq_neon_slow_block(const GptqBlockView &bv, const float *x, float *y,
                                 int out_dim, int in_dim, int group_size,
                                 int ob, int blen) {
    const int n_c8 = in_dim / kGptqPackInts;
    const uint8_t *qw = bv.qweight;
    const uint8_t *sc = bv.scales;
    const uint8_t *qz = bv.qzeros;
    const uint8_t *gi = bv.g_idx;

    for (int o = 0; o < blen; ++o) {
        float a = 0.0f;
        for (int c8 = 0; c8 < n_c8; ++c8) {
            const uint32_t word = gptq_read_u32(
                qw + (static_cast<size_t>(c8) * out_dim + ob + o) * 4);
            const float *x8 = x + c8 * kGptqPackInts;
            for (int k = 0; k < kGptqPackInts; ++k) {
                const uint32_t nib = (word >> (k * 4)) & 0xFu;
                const int g = gi ? static_cast<int>(
                                       gptq_read_u32(gi + static_cast<size_t>(c8 * 8 + k) * 4))
                                 : (c8 * 8 + k) / group_size;
                uint16_t sh, zh;
                std::memcpy(&sh,
                            sc + (static_cast<size_t>(g) * out_dim + ob + o) * 2, 2);
                std::memcpy(&zh,
                            qz + (static_cast<size_t>(g) * out_dim + ob + o) * 2, 2);
                const float s =
                    static_cast<float>(*reinterpret_cast<const float16_t *>(&sh));
                const float z =
                    static_cast<float>(*reinterpret_cast<const float16_t *>(&zh));
                a += (static_cast<float>(nib) - z) * s * x8[k];
            }
        }
        y[ob + o] = a;
    }
}

// 预算 sx8[c8] = Σ_{k=0..7} x[c8*8+k]，以及每个 c8 的组号。
// 两者都只依赖 c8，与 o 无关 —— 这是因式分解能省掉 8 倍 s/z 运算的前提。
inline void gptq_precompute(const float *x, int in_dim, int group_size,
                            const GptqBlockView &bv, float *sx8, int *g_of_c8) {
    const int n_c8 = in_dim / kGptqPackInts;
    for (int c8 = 0; c8 < n_c8; ++c8) {
        const float *x8 = x + c8 * kGptqPackInts;
        sx8[c8] = x8[0] + x8[1] + x8[2] + x8[3] + x8[4] + x8[5] + x8[6] + x8[7];
        g_of_c8[c8] = bv.g_idx
            ? static_cast<int>(gptq_read_u32(bv.g_idx + static_cast<size_t>(c8 * 8) * 4))
            : (c8 * kGptqPackInts) / group_size;
    }
}

} // namespace
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
