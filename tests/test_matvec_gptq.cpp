// ============================================================================
// test_matvec_gptq.cpp — 原生 GPTQ-INT4 matvec 单元测试
// ============================================================================
// 验证 matvec_gptq_ref 的 AutoGPTQ 列主序 dequant + matvec 正确性。
//
// 测试方法论（与 test_matvec_i4 同款，独立实现 pack + naive）：
//   1. pack_gptq()      — 测试侧独立 RTN 量化 + AutoGPTQ 列主序 int32 打包
//   2. naive_matvec_gptq() — 测试侧独立 dequant + double 累加 matvec
//   3. 对比 matvec_gptq (ref) 输出与 naive 输出，tol 1e-3
//
// 覆盖 has_g_idx=true（contiguous g_idx）与 has_g_idx=false 两种布局，
// 二者反量化结果必须一致（g_idx 只是显式给列→组映射）。
// ============================================================================

#include "test_framework.h"

#include "dispatch.h"   // set_matvec_gptq_impl_by_name / matvec_gptq
#include "ref_ops.h"    // float_to_half / half_to_float

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

using tinyqwen::float_to_half;
using tinyqwen::half_to_float;

constexpr int kHdr = 8;       // GPTQ in-band header: {u32 magic, u32 flags}
constexpr uint32_t kGptqMagic = 0x47505451u;
constexpr int kPack = 8;      // 每个 int32 装 8 个 uint4

// 镜像 matvec_gptq_ref 的列主序 qword 布局：qweight[c8 * out_dim + o]，
// word 的第 k 个 nibble（>>k*4）对应 input col = c8*8 + k、output row = o。
std::vector<uint8_t> pack_gptq(const std::vector<float> &w, int out_dim, int in_dim,
                                int group_size, bool has_g_idx) {
    const int n_groups = in_dim / group_size;
    const size_t sq = static_cast<size_t>(n_groups) * out_dim * 2;
    const size_t gidx_bytes = has_g_idx ? static_cast<size_t>(in_dim) * 4 : 0;
    const size_t qw_bytes = static_cast<size_t>(in_dim / kPack) * out_dim * 4;
    std::vector<uint8_t> buf(kHdr + sq + sq + gidx_bytes + qw_bytes, 0);

    // header
    std::memcpy(buf.data(), &kGptqMagic, 4);
    uint32_t flags = has_g_idx ? 1u : 0u;
    std::memcpy(buf.data() + 4, &flags, 4);

    uint8_t *scales = buf.data() + kHdr;
    uint8_t *qzeros = scales + sq;
    uint8_t *g_idx = has_g_idx ? (qzeros + sq) : nullptr;
    uint8_t *qweight = has_g_idx ? (g_idx + gidx_bytes) : (qzeros + sq);

    // RTN 量化：每 (group g, output row o) 一组 scale/zero（列方向分组）
    std::vector<int> q(static_cast<size_t>(out_dim) * in_dim, 0);
    for (int g = 0; g < n_groups; ++g) {
        for (int o = 0; o < out_dim; ++o) {
            float vmin = w[o * in_dim + g * group_size];
            float vmax = vmin;
            for (int i = 0; i < group_size; ++i) {
                const float v = w[o * in_dim + g * group_size + i];
                vmin = std::min(vmin, v);
                vmax = std::max(vmax, v);
            }
            const float scale = (vmax > vmin) ? (vmax - vmin) / 15.0f : 0.0f;
            const float zero = (scale > 0) ? -vmin / scale : 0.0f;
            const uint16_t sh = float_to_half(scale);
            const uint16_t zh = float_to_half(zero);
            const float s = half_to_float(sh);
            const float z = half_to_float(zh);
            std::memcpy(scales + (static_cast<size_t>(g) * out_dim + o) * 2, &sh, 2);
            std::memcpy(qzeros + (static_cast<size_t>(g) * out_dim + o) * 2, &zh, 2);
            for (int i = 0; i < group_size; ++i) {
                int qq = (s > 0) ? static_cast<int>(std::lround(
                            w[o * in_dim + g * group_size + i] / s + z)) : 0;
                qq = std::max(0, std::min(15, qq));
                q[static_cast<size_t>(o) * in_dim + g * group_size + i] = qq;
            }
        }
    }

    // 打包 qweight：qweight[c8 * out_dim + o] 的第 k nibble = q[o][c8*8+k]
    for (int c8 = 0; c8 < in_dim / kPack; ++c8) {
        for (int o = 0; o < out_dim; ++o) {
            uint32_t word = 0;
            for (int k = 0; k < kPack; ++k) {
                const int c = c8 * kPack + k;
                const uint32_t qq = static_cast<uint32_t>(
                    q[static_cast<size_t>(o) * in_dim + c]);
                word |= (qq & 0xFu) << (k * 4);
            }
            std::memcpy(qweight + (static_cast<size_t>(c8) * out_dim + o) * 4, &word, 4);
        }
    }

    // g_idx：contiguous 组映射 g_idx[c] = c / group_size
    if (has_g_idx) {
        for (int c = 0; c < in_dim; ++c) {
            const uint32_t gv = static_cast<uint32_t>(c / group_size);
            std::memcpy(g_idx + static_cast<size_t>(c) * 4, &gv, 4);
        }
    }
    return buf;
}

// 独立 dequant + double 累加 matvec（与 matvec_gptq_ref 同语义）
std::vector<float> naive_matvec_gptq(const std::vector<uint8_t> &buf,
                                     const std::vector<float> &x,
                                     int out_dim, int in_dim, int group_size) {
    const int n_groups = in_dim / group_size;
    const size_t sq = static_cast<size_t>(n_groups) * out_dim * 2;
    uint32_t flags;
    std::memcpy(&flags, buf.data() + 4, 4);
    const bool has_g_idx = (flags & 1u) != 0;
    const uint8_t *scales = buf.data() + kHdr;
    const uint8_t *qzeros = scales + sq;
    const uint8_t *g_idx = has_g_idx ? (qzeros + sq) : nullptr;
    const uint8_t *qweight = has_g_idx ? (g_idx + static_cast<size_t>(in_dim) * 4)
                                       : (qzeros + sq);

    std::vector<float> y(out_dim, 0.0f);
    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        for (int c8 = 0; c8 < in_dim / kPack; ++c8) {
            uint32_t word;
            std::memcpy(&word, qweight + (static_cast<size_t>(c8) * out_dim + o) * 4, 4);
            for (int k = 0; k < kPack; ++k) {
                const int c = c8 * kPack + k;
                const uint32_t nibble = (word >> (k * 4)) & 0xFu;
                int g;
                if (has_g_idx) {
                    std::memcpy(&g, g_idx + static_cast<size_t>(c) * 4, 4);
                } else {
                    g = c / group_size;
                }
                uint16_t sh, zh;
                std::memcpy(&sh, scales + (static_cast<size_t>(g) * out_dim + o) * 2, 2);
                std::memcpy(&zh, qzeros + (static_cast<size_t>(g) * out_dim + o) * 2, 2);
                const float s = half_to_float(sh);
                const float z = half_to_float(zh);
                acc += static_cast<double>((static_cast<float>(nibble) - z) * s) *
                       static_cast<double>(x[c]);
            }
        }
        y[o] = static_cast<float>(acc);
    }
    return y;
}

std::vector<float> random_vec(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

void check_gptq(int out_dim, int in_dim, int group_size, bool has_g_idx) {
    EXPECT_TRUE(tinyqwen::set_matvec_gptq_impl_by_name("ref"));
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 42);
    const std::vector<float> x = random_vec(in_dim, 7);
    const std::vector<uint8_t> buf = pack_gptq(w, out_dim, in_dim, group_size, has_g_idx);
    std::vector<float> y(out_dim, 0.0f);
    tinyqwen::matvec_gptq(buf.data(), x.data(), y.data(), out_dim, in_dim, group_size);
    const std::vector<float> ref = naive_matvec_gptq(buf, x, out_dim, in_dim, group_size);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i)
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    if (max_err >= 1e-3) {
        TQ_FAIL("matvec_gptq max_err=" + std::to_string(max_err) +
                " (out=" + std::to_string(out_dim) + " in=" + std::to_string(in_dim) +
                " gs=" + std::to_string(group_size) + " gidx=" +
                std::to_string(has_g_idx ? 1 : 0) + ")");
    }
}

} // namespace

TEST(matvec_gptq_ref_no_gidx) {
    check_gptq(16, 32, 16, false);
    check_gptq(32, 64, 32, false);
    check_gptq(48, 128, 32, false); // 多组
}

TEST(matvec_gptq_ref_with_gidx) {
    // has_g_idx=true（contiguous g_idx）必须与 has_g_idx=false 逐位一致
    check_gptq(16, 32, 16, true);
    check_gptq(32, 64, 32, true);
    check_gptq(48, 128, 32, true);
}

TEST(matvec_gptq_gidx_equivalence) {
    // 同一份权重，两种布局的 matvec 结果应一致（g_idx 只是显式映射）
    const int out_dim = 24, in_dim = 64, gs = 16;
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 99);
    const std::vector<float> x = random_vec(in_dim, 5);
    const std::vector<uint8_t> b_no = pack_gptq(w, out_dim, in_dim, gs, false);
    const std::vector<uint8_t> b_gi = pack_gptq(w, out_dim, in_dim, gs, true);
    std::vector<float> y_no(out_dim), y_gi(out_dim);
    tinyqwen::matvec_gptq(b_no.data(), x.data(), y_no.data(), out_dim, in_dim, gs);
    tinyqwen::matvec_gptq(b_gi.data(), x.data(), y_gi.data(), out_dim, in_dim, gs);
    for (int i = 0; i < out_dim; ++i)
        EXPECT_NEAR(y_no[i], y_gi[i], 1e-5);
}
