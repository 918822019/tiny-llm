// INT4 matvec / matmul 单元测试。
//
// 背景：i4 kernel 此前只有 ref 注释约束、没有自动化验证。量化导出（RTN/HQQ）
// 与 kernel 之间的契约（interleaved 布局、低 nibble 在前、(q-zero)×scale 反量化、
// fp16 scale/zero）全靠这些测试把关。
//
// kernel 函数注册在匿名 namespace，测试统一走 dispatch（set_matvec_i4_impl_by_name）。

#include "test_framework.h"

#include "dispatch.h"
#include "ref_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace {

using tinyqwen::float_to_half;
using tinyqwen::half_to_float;

constexpr int kHdr = 4; // scale(fp16) + zero(fp16)

// RTN 量化并 pack 成 interleaved i4 字节（镜像 exporter 的 quantize_tensor_i4）。
std::vector<uint8_t> pack_i4_rtn(const std::vector<float> &w, int out_dim, int in_dim,
                                 int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    std::vector<uint8_t> packed(static_cast<size_t>(out_dim) * row_bytes, 0);
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            float vmin = w[o * in_dim + start], vmax = vmin;
            for (int i = start; i < end; ++i) {
                vmin = std::min(vmin, w[o * in_dim + i]);
                vmax = std::max(vmax, w[o * in_dim + i]);
            }
            const float scale = (vmax > vmin) ? (vmax - vmin) / 15.0f : 0.0f;
            const float zero = (scale > 0) ? -vmin / scale : 0.0f;
            const uint16_t scale_h = float_to_half(scale);
            const uint16_t zero_h = float_to_half(zero);
            // 与 C++ kernel 对齐：用 fp16 往返后的 s/z 算 q
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            std::memcpy(gp, &scale_h, 2);
            std::memcpy(gp + 2, &zero_h, 2);
            for (int i = start; i < end; ++i) {
                int q = (s > 0) ? static_cast<int>(std::lround(w[o * in_dim + i] / s + z)) : 0;
                q = std::max(0, std::min(15, q));
                const int li = i - start;
                if (li % 2 == 0) gp[kHdr + li / 2] = static_cast<uint8_t>(q);
                else gp[kHdr + li / 2] |= static_cast<uint8_t>(q << 4);
            }
        }
    }
    return packed;
}

// 测试侧独立实现：从 packed 字节反量化 + 矩阵向量乘（double 累加）。
std::vector<float> naive_matvec_i4(const std::vector<uint8_t> &packed,
                                   const std::vector<float> &x,
                                   int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            for (int i = start; i < end; ++i) {
                const int li = i - start;
                const uint8_t byte = gp[kHdr + li / 2];
                const int q = (li % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                acc += static_cast<double>((q - z) * s) * x[i];
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
    for (auto &x: v) x = dist(rng);
    return v;
}

void check_matvec_i4_impl(const char *impl, int out_dim, int in_dim, int group_size) {
    if (!tinyqwen::set_matvec_i4_impl_by_name(impl)) {
        // 非 aarch64 平台没有 neon 系实现——跳过而不是失败
        if (std::string(impl).rfind("neon", 0) == 0) return;
        TQ_FAIL(std::string("set_matvec_i4_impl_by_name failed: ") + impl);
    }
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 42);
    const std::vector<float> x = random_vec(in_dim, 7);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> y(out_dim);
    tinyqwen::matvec_i4(packed.data(), x.data(), y.data(), out_dim, in_dim, group_size);
    const std::vector<float> ref = naive_matvec_i4(packed, x, out_dim, in_dim, group_size);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    // ref kernel 用 double 累加、naive 也是 double：差异只来自 float 舍入
    if (max_err >= 1e-3) {
        TQ_FAIL("matvec_i4(" + std::string(impl) + ", g=" + std::to_string(group_size) +
                ") max_err=" + std::to_string(max_err));
    }
}

} // namespace

// 手工小例：逐字节构造权重，独立手算期望值——验证布局约定本身
// （header 顺序、低 nibble 在前、(q-zero)×scale、fp16 scale/zero）。
TEST(matvec_i4_handcrafted_layout) {
    // 1 行 × 4 元素，group_size=4（1 组）：q = [1, 2, 15, 0]
    // scale=2.0, zero=1.0 → dequant = [0, 2, 28, -2]；x = [1,1,1,1] → y = 28
    const uint16_t scale_h = float_to_half(2.0f);
    const uint16_t zero_h = float_to_half(1.0f);
    std::vector<uint8_t> packed(kHdr + 2, 0);
    std::memcpy(packed.data(), &scale_h, 2);
    std::memcpy(packed.data() + 2, &zero_h, 2);
    packed[kHdr] = (2 << 4) | 1;  // byte0: 元素1=2（高 nibble）, 元素0=1（低）
    packed[kHdr + 1] = (0 << 4) | 15; // byte1: 元素3=0（高）, 元素2=15（低）
    const float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y = 0.0f;
    tinyqwen::set_matvec_i4_impl_by_name("ref");
    tinyqwen::matvec_i4(packed.data(), x, &y, 1, 4, 4);
    EXPECT_NEAR(y, 28.0f, 1e-5);
}

// ref kernel vs 测试侧独立实现（随机权重）
TEST(matvec_i4_ref_matches_naive_g128) { check_matvec_i4_impl("ref", 64, 256, 128); }
TEST(matvec_i4_ref_matches_naive_g64) { check_matvec_i4_impl("ref", 64, 256, 64); }
// in_dim 非 group 整数倍（尾部组）也要对
TEST(matvec_i4_ref_partial_group) { check_matvec_i4_impl("ref", 32, 200, 64); }

// NEON kernel 与 ref 对齐（仅 aarch64 有注册；其他平台自动跳过）
TEST(matvec_i4_neon_matches_naive_g128) { check_matvec_i4_impl("neon", 96, 384, 128); }
TEST(matvec_i4_neon_matches_naive_g64) { check_matvec_i4_impl("neon", 96, 384, 64); }

// NEON + 多线程：行切分不改变数值，结果必须与单线程一致。
// 用 > kMinParallelElems（262144）的矩阵确保真的走多线程路径。
TEST(matvec_i4_neon_mt_matches_naive_g64) { check_matvec_i4_impl("neon_mt", 512, 896, 64); }
TEST(matvec_i4_neon_mt_matches_naive_g128) { check_matvec_i4_impl("neon_mt", 512, 1024, 128); }
// NEON + fp16 LUT：查表反量化 + f16 点积，数值必须与朴素反量化一致
// （组末 flush fp32，组内 f16 累加误差受限）。
// W4A8 SDOT 的朴素参考：与 kernel 同一套量化方案（激活对称 int8 + 非对称 zero
// 修正），用纯标量整数点积实现，用来核对 NEON SDOT kernel 的正确性。
// 两者同方案，应几乎逐位一致（只有 fp32 加法顺序差）。
std::vector<float> matvec_i4_w4a8_naive(const std::vector<uint8_t> &packed,
                                        const std::vector<float> &x,
                                        int out_dim, int in_dim, int group_size) {
    // 激活对称 int8 量化（与 kernel 的 quantize_x_i8 逐字一致）
    float amax = 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale_x;
    std::vector<int> xq(in_dim);
    std::vector<int> prefix(in_dim + 1, 0);
    for (int i = 0; i < in_dim; ++i) {
        float v = x[i] * inv;
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);
        xq[i] = q;
        prefix[i + 1] = prefix[i] + q;
    }

    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = packed.data() + static_cast<size_t>(o) * row_bytes;
        float acc = 0.0f;
        int col = 0;
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = row + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float scale_w = tinyqwen::half_to_float(scale_h);
            const float zero_w = tinyqwen::half_to_float(zero_h);
            const float A = scale_w * scale_x;
            const float C = A * (zero_w - 8.0f);
            const uint8_t *pk = gp + kHdr;
            const int elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);
            int dot_g = 0;
            for (int i = 0; i < elems; ++i) {
                const int q = (i % 2 == 0) ? (pk[i / 2] & 0x0F) : ((pk[i / 2] >> 4) & 0x0F);
                dot_g += (q - 8) * xq[col + i];
            }
            const int xqsum = prefix[col + elems] - prefix[col];
            acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum);
            col += group_size;
        }
        y[o] = acc;
    }
    return y;
}

// SDOT kernel vs 朴素 W4A8（同方案，应几乎逐位一致）
void check_matvec_i4_sdot_impl(const char *impl, int out_dim, int in_dim, int group_size) {
    if (!tinyqwen::set_matvec_i4_impl_by_name(impl)) {
        return; // 平台无 dotprod，跳过
    }
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 42);
    const std::vector<float> x = random_vec(in_dim, 7);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> y(out_dim);
    tinyqwen::matvec_i4(packed.data(), x.data(), y.data(), out_dim, in_dim, group_size);
    const std::vector<float> ref = matvec_i4_w4a8_naive(packed, x, out_dim, in_dim, group_size);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    if (max_err >= 1e-2) {
        TQ_FAIL(std::string("matvec_i4(") + impl + ", g=" + std::to_string(group_size) +
                ") max_err=" + std::to_string(max_err));
    }
}
void check_matvec_i4_sdot(int out_dim, int in_dim, int group_size) {
    check_matvec_i4_sdot_impl("sdot", out_dim, in_dim, group_size);
}
TEST(matvec_i4_sdot_matches_w4a8_naive_g64) { check_matvec_i4_sdot(96, 384, 64); }
TEST(matvec_i4_sdot_matches_w4a8_naive_g128) { check_matvec_i4_sdot(96, 384, 128); }
TEST(matvec_i4_sdot_partial) { check_matvec_i4_sdot(32, 200, 64); }
TEST(matvec_i4_sdot_mt_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot_mt", 512, 896, 64); }
TEST(matvec_i4_sdot_mt_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot_mt", 512, 1024, 128); }

// 小矩阵走单线程分支也要对
TEST(matvec_i4_neon_mt_small_singlethread) { check_matvec_i4_impl("neon_mt", 32, 64, 64); }

// matmul_i4（prefill 路径）：N=2 列主序输入，每列应等于对应 matvec
TEST(matmul_i4_matches_matvec) {
    tinyqwen::set_matvec_i4_impl_by_name("ref");
    const int M = 48, K = 128, N = 2, G = 64;
    const std::vector<float> w = random_vec(static_cast<size_t>(M) * K, 99);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, M, K, G);
    const std::vector<float> x0 = random_vec(K, 1);
    const std::vector<float> x1 = random_vec(K, 2);
    // col-major [K, N]
    std::vector<float> xKN(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) { xKN[k] = x0[k]; xKN[K + k] = x1[k]; }
    std::vector<float> yMN(static_cast<size_t>(M) * N);
    tinyqwen::matmul_i4(packed.data(), xKN.data(), yMN.data(), M, K, N, G);
    std::vector<float> y0(M), y1(M);
    tinyqwen::matvec_i4(packed.data(), x0.data(), y0.data(), M, K, G);
    tinyqwen::matvec_i4(packed.data(), x1.data(), y1.data(), M, K, G);
    for (int m = 0; m < M; ++m) {
        EXPECT_NEAR(yMN[m], y0[m], 1e-4);
        EXPECT_NEAR(yMN[M + m], y1[m], 1e-4);
    }
}

// dispatch：未知名 fail fast
TEST(matvec_i4_dispatch_unknown_fails) {
    EXPECT_TRUE(!tinyqwen::set_matvec_i4_impl_by_name("no_such_impl"));
    EXPECT_TRUE(tinyqwen::set_matvec_i4_impl_by_name("ref"));
    EXPECT_EQ(std::string(tinyqwen::matvec_i4_impl_name()), std::string("ref"));
}
