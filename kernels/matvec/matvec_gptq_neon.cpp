// ============================================================================
// matvec_gptq_neon.cpp — GPTQ-INT4 matvec NEON 变体（归因阶梯 L2）
// ============================================================================
// 相对 matvec_gptq_ref 的三处改动，各自独立可归因：
//   ① 遍历顺序 c8-outer / o-inner（ref 是 o-outer / c8-inner）。qweight 是
//      [in_dim/8, out_dim]，固定 c8 变化 o 才是连续内存；ref 的顺序按
//      out_dim*4 字节跨步，cache 命中极差。
//   ② o 方向分块 64：每块 256 B = 4 条完整 cache line，且累加器留在寄存器。
//   ③ 反量化因式分解：同一 u32 字的 8 个 nibble 共享 s/z，故
//      Σ_k ((nib_k - z)·s·x_k) = s·[Σ_k(nib_k·x_k) - z·Σ_k(x_k)]，
//      Σ_k(x_k) 只依赖 c8、与 o 无关，可预算。每 8 权重省掉 8 次减+乘。
//
// 正确性判据：与 matvec_gptq_ref（double 累加）容差对齐，容差实测值见
// tests/test_gptq_matvec.cpp。act-order（g_idx 非均匀）走 slow 路径，
// 不做因式分解——算错不报错是本仓库最忌讳的失效模式。
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include "matvec_gptq_neon_common.h"

#include <vector>

namespace tinyqwen {
namespace {

void matvec_gptq_neon(const uint8_t *w, const float *x, float *y, int out_dim,
                      int in_dim, int group_size) {
    // ref 的前置条件：in_dim 须同时被 8（打包）和 group_size（分组）整除
    if (in_dim % kGptqPackInts != 0 || in_dim % group_size != 0) {
        matvec_gptq_ref(w, x, y, out_dim, in_dim, group_size);
        return;
    }

    const GptqBlockView bv = parse_gptq_block(w, out_dim, in_dim, group_size);
    const int n_c8 = in_dim / kGptqPackInts;

    std::vector<float> sx8(static_cast<size_t>(n_c8));
    std::vector<int> g_of_c8(static_cast<size_t>(n_c8));
    gptq_precompute(x, in_dim, group_size, bv, sx8.data(), g_of_c8.data());

    // act-order GPTQ 的 g_idx 是排列，8 行可能跨组 → 因式分解不成立
    const bool uniform = gptq_g_idx_uniform(bv.g_idx, in_dim);

    for (int ob = 0; ob < out_dim; ob += kGptqOBlock) {
        const int blen = (out_dim - ob < kGptqOBlock) ? out_dim - ob : kGptqOBlock;
        if (uniform) {
            gptq_neon_block(bv, x, y, sx8.data(), g_of_c8.data(), out_dim, in_dim,
                            ob, blen);
        } else {
            gptq_neon_slow_block(bv, x, y, out_dim, in_dim, group_size, ob, blen);
        }
    }
}

} // namespace

TINYQWEN_MATVEC_GPTQ_VARIANT(matvec_gptq_neon, "neon");

} // namespace tinyqwen

#else // 非 ARM：兜底到 ref，保证任何平台都能编译并自注册

namespace tinyqwen {
namespace {

void matvec_gptq_neon(const uint8_t *w, const float *x, float *y, int out_dim,
                      int in_dim, int group_size) {
    matvec_gptq_ref(w, x, y, out_dim, in_dim, group_size);
}

} // namespace

TINYQWEN_MATVEC_GPTQ_VARIANT(matvec_gptq_neon, "neon");

} // namespace tinyqwen

#endif
