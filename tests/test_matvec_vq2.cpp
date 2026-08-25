// ============================================================================
// test_matvec_vq2.cpp — VQ2（2-bit 块向量量化）matvec / pair / qkv / matmul 单元测试
// ============================================================================
// 验证 VQ2 块向量量化矩阵乘法的正确性，覆盖：
//   - ref  — 标量参考实现（double 累加，纯查表 gather）
//   - neon — 4 宽 NEON FMA（仅 aarch64）
//
// VQ2 布局（与 runtime/tiny_format.h 的 kVQ2* 常量一致）：
//   [ 码本: [K=256, d=4] fp16 = 2048B ][ 索引: out_dim × (in_dim/4) × uint8 ]
//   block b 的 d=4 个连续权重 = codebook[index_b]（一个 4 维向量）。
//   码率 = log2(256)/4 = 2 bit/权重；反量化纯查表，零算术。
//
// 测试方法论（与 test_matvec_i4.cpp 同思路）：
//   1. pack_vq2()        — 测试侧独立打包（给定码本 + 索引）
//   2. naive_matvec_vq2()— 测试侧独立反量化 + double 累加 matvec
//   3. check_*()         — 将 kernel 输出与独立实现对比
//
// 注意：内核正确性不依赖 k-means——给定任意码本 + 索引，内核都应正确地
// "查表 + 点积"。因此这里用随机码本 + 随机索引直接构造，隔离测试内核契约。
// 要求 in_dim 被 4 整除（与加载器/导出器约束一致）。
// ============================================================================

#include "test_framework.h"  // 自研测试框架

#include "dispatch.h"   // matvec_vq2 / set_matvec_vq2_impl_by_name / pair / qkv / matmul
#include "ref_ops.h"    // float_to_half / half_to_float

#include <cmath>        // std::fabs
#include <cstdint>      // uint8_t, uint16_t
#include <cstring>      // std::memcpy
#include <random>       // std::mt19937
#include <string>
#include <vector>

namespace {

using tinyqwen::float_to_half;
using tinyqwen::half_to_float;

constexpr int kD = 4;               // 块大小
constexpr int kK = 256;             // 码本条目数
constexpr int kCbBytes = kK * kD * 2; // 码本字节数 = 2048B

// -------------------------------------------------------------------------
// random_vec() — 生成正态分布随机向量
// -------------------------------------------------------------------------
std::vector<float> random_vec(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

// -------------------------------------------------------------------------
// pack_vq2() — 把给定 fp16 码本 [K, d] + uint8 索引打包成 [码本][索引]
// -------------------------------------------------------------------------
std::vector<uint8_t> pack_vq2(const std::vector<uint16_t> &codebook_h, // K*d
                              const std::vector<uint8_t> &indices,      // out_dim*(in_dim/d)
                              int out_dim, int in_dim) {
    const int n_blocks = in_dim / kD;
    std::vector<uint8_t> blob(kCbBytes + static_cast<size_t>(out_dim) * n_blocks);
    for (int i = 0; i < kK * kD; ++i) {
        std::memcpy(blob.data() + i * 2, &codebook_h[i], 2);
    }
    std::memcpy(blob.data() + kCbBytes, indices.data(),
                static_cast<size_t>(out_dim) * n_blocks);
    return blob;
}

// -------------------------------------------------------------------------
// naive_matvec_vq2() — 独立反量化 + double 累加 matvec（正确性锚）
// -------------------------------------------------------------------------
std::vector<float> naive_matvec_vq2(const std::vector<uint8_t> &blob,
                                    const std::vector<float> &x,
                                    int out_dim, int in_dim) {
    float cb[kK][kD];
    for (int k = 0; k < kK; ++k) {
        for (int j = 0; j < kD; ++j) {
            uint16_t h;
            std::memcpy(&h, blob.data() + (k * kD + j) * 2, 2);
            cb[k][j] = half_to_float(h);
        }
    }
    const int n_blocks = in_dim / kD;
    const uint8_t *idx = blob.data() + kCbBytes;
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        double acc = 0.0;
        for (int b = 0; b < n_blocks; ++b) {
            const float *cv = cb[row[b]];
            const int base = b * kD;
            for (int j = 0; j < kD; ++j) {
                acc += static_cast<double>(cv[j]) * static_cast<double>(x[base + j]);
            }
        }
        y[o] = static_cast<float>(acc);
    }
    return y;
}

// -------------------------------------------------------------------------
// make_random_vq2() — 随机码本 + 随机索引（构造被测张量）
// -------------------------------------------------------------------------
struct VQ2Case {
    std::vector<uint8_t> blob;
    std::vector<float> x;
};
VQ2Case make_random_vq2(int out_dim, int in_dim) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> val_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> idx_dist(0, kK - 1);
    std::vector<uint16_t> cb_h(kK * kD);
    for (auto &h : cb_h) h = float_to_half(val_dist(rng));
    const int n_blocks = in_dim / kD;
    std::vector<uint8_t> indices(static_cast<size_t>(out_dim) * n_blocks);
    for (auto &b : indices) b = static_cast<uint8_t>(idx_dist(rng));
    VQ2Case c;
    c.blob = pack_vq2(cb_h, indices, out_dim, in_dim);
    c.x = random_vec(in_dim, 7);
    return c;
}

// -------------------------------------------------------------------------
// check_matvec_vq2_impl() — 通用 matvec_vq2 正确性检查
// -------------------------------------------------------------------------
void check_matvec_vq2_impl(const char *impl, int out_dim, int in_dim, double tol) {
    if (!tinyqwen::set_matvec_vq2_impl_by_name(impl)) {
        if (std::string(impl).rfind("neon", 0) == 0) return; // 非 aarch64 跳过
        TQ_FAIL(std::string("set_matvec_vq2_impl_by_name failed: ") + impl);
    }
    const VQ2Case c = make_random_vq2(out_dim, in_dim);
    std::vector<float> y(out_dim);
    tinyqwen::matvec_vq2(c.blob.data(), c.x.data(), y.data(), out_dim, in_dim);
    const std::vector<float> ref = naive_matvec_vq2(c.blob, c.x, out_dim, in_dim);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    if (max_err >= tol) {
        TQ_FAIL("matvec_vq2(" + std::string(impl) + ") max_err=" + std::to_string(max_err));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// 手工小例验证 VQ2 块布局约定
//
// 1 行 × 4 元素 = 1 块；码本 cb[5] = [5,5,5,5]；索引 = 5；x = [1,1,1,1]
//   y = cb[5]·x = 5+5+5+5 = 20
// ---------------------------------------------------------------------------
TEST(matvec_vq2_handcrafted_layout) {
    std::vector<uint16_t> cb_h(kK * kD, 0);
    for (int j = 0; j < kD; ++j) cb_h[5 * kD + j] = float_to_half(5.0f);
    const std::vector<uint8_t> indices = {5};   // 1 行 1 块
    const std::vector<uint8_t> blob = pack_vq2(cb_h, indices, 1, 4);
    const float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y = 0.0f;
    tinyqwen::set_matvec_vq2_impl_by_name("ref");
    tinyqwen::matvec_vq2(blob.data(), x, &y, 1, 4);
    EXPECT_NEAR(y, 20.0f, 1e-5);
}

// ---------------------------------------------------------------------------
// 未注册的实现名必须返回 false（fail-fast，不能静默回退）
// ---------------------------------------------------------------------------
TEST(matvec_vq2_unknown_impl_fails) {
    EXPECT_TRUE(!tinyqwen::set_matvec_vq2_impl_by_name("definitely_not_a_real_impl"));
}

// ---------------------------------------------------------------------------
// ref kernel vs naive 独立实现（随机码本 + 随机索引；in_dim 均被 4 整除）
// ---------------------------------------------------------------------------
TEST(matvec_vq2_ref_matches_naive_small) { check_matvec_vq2_impl("ref", 32, 128, 1e-3); }
TEST(matvec_vq2_ref_matches_naive_mid)   { check_matvec_vq2_impl("ref", 96, 384, 1e-3); }
TEST(matvec_vq2_ref_matches_naive_odd)   { check_matvec_vq2_impl("ref", 33, 200, 1e-3); }

// ---------------------------------------------------------------------------
// neon kernel（fp32 累加，容差放宽）；非 aarch64 自动跳过
// ---------------------------------------------------------------------------
TEST(matvec_vq2_neon_matches_naive_small) { check_matvec_vq2_impl("neon", 32, 128, 1e-2); }
TEST(matvec_vq2_neon_matches_naive_mid)   { check_matvec_vq2_impl("neon", 96, 384, 1e-2); }
TEST(matvec_vq2_neon_matches_naive_big)   { check_matvec_vq2_impl("neon", 256, 896, 1e-2); }

// neon_mr（4 行并行）：与 naive 对齐；含非 4 倍数行数（尾部单行路径）
TEST(matvec_vq2_neon_mr_matches_naive_small) { check_matvec_vq2_impl("neon_mr", 32, 128, 1e-2); }
TEST(matvec_vq2_neon_mr_matches_naive_mid)   { check_matvec_vq2_impl("neon_mr", 96, 384, 1e-2); }
TEST(matvec_vq2_neon_mr_matches_naive_big)   { check_matvec_vq2_impl("neon_mr", 256, 896, 1e-2); }
TEST(matvec_vq2_neon_mr_matches_naive_tail)  { check_matvec_vq2_impl("neon_mr", 33, 200, 1e-2); }

// neon_mr_mt（4 行并行 + 线程池行切分）：小形状走单线程路径，
// ≥262144 元素走多线程路径；513 行带区间尾行
TEST(matvec_vq2_neon_mr_mt_matches_naive_small) { check_matvec_vq2_impl("neon_mr_mt", 32, 128, 1e-2); }
TEST(matvec_vq2_neon_mr_mt_matches_naive_mt)    { check_matvec_vq2_impl("neon_mr_mt", 512, 896, 1e-2); }
TEST(matvec_vq2_neon_mr_mt_matches_naive_tail)  { check_matvec_vq2_impl("neon_mr_mt", 513, 896, 1e-2); }

// ---------------------------------------------------------------------------
// pair / qkv 融合入口（v1 拆成多次 matvec_vq2）结果须与独立 naive 一致
// ---------------------------------------------------------------------------
TEST(matvec_vq2_pair_decomposes_correctly) {
    tinyqwen::set_matvec_vq2_impl_by_name("ref");
    const int D = 64, IN = 256;
    const VQ2Case a = make_random_vq2(D, IN);
    const VQ2Case b = make_random_vq2(D, IN);
    std::vector<float> y1(D), y2(D);
    tinyqwen::matvec_pair_vq2(a.blob.data(), b.blob.data(), a.x.data(),
                              y1.data(), y2.data(), D, IN);
    const std::vector<float> r1 = naive_matvec_vq2(a.blob, a.x, D, IN);
    const std::vector<float> r2 = naive_matvec_vq2(b.blob, a.x, D, IN);
    double e = 0.0;
    for (int i = 0; i < D; ++i) {
        e = std::max(e, std::fabs(static_cast<double>(y1[i]) - r1[i]));
        e = std::max(e, std::fabs(static_cast<double>(y2[i]) - r2[i]));
    }
    EXPECT_TRUE(e < 1e-3);
}

TEST(matvec_vq2_qkv_decomposes_correctly) {
    tinyqwen::set_matvec_vq2_impl_by_name("ref");
    const int QD = 128, KVD = 32, IN = 256;
    const VQ2Case q = make_random_vq2(QD, IN);
    const VQ2Case k = make_random_vq2(KVD, IN);
    const VQ2Case v = make_random_vq2(KVD, IN);
    std::vector<float> yq(QD), yk(KVD), yv(KVD);
    tinyqwen::matvec_qkv_vq2(q.blob.data(), k.blob.data(), v.blob.data(), q.x.data(),
                             yq.data(), yk.data(), yv.data(), QD, KVD, IN);
    const std::vector<float> rq = naive_matvec_vq2(q.blob, q.x, QD, IN);
    const std::vector<float> rk = naive_matvec_vq2(k.blob, q.x, KVD, IN);
    const std::vector<float> rv = naive_matvec_vq2(v.blob, q.x, KVD, IN);
    double e = 0.0;
    for (int i = 0; i < QD; ++i) e = std::max(e, std::fabs((double) yq[i] - rq[i]));
    for (int i = 0; i < KVD; ++i) {
        e = std::max(e, std::fabs((double) yk[i] - rk[i]));
        e = std::max(e, std::fabs((double) yv[i] - rv[i]));
    }
    EXPECT_TRUE(e < 1e-3);
}

// ---------------------------------------------------------------------------
// matmul（prefill，列主序 X/Y）：逐列须等于 matvec_vq2
// ---------------------------------------------------------------------------
TEST(matvec_vq2_matmul_columns_match_matvec) {
    tinyqwen::set_matvec_vq2_impl_by_name("ref");
    const int M = 48, K = 192, N = 5;
    const VQ2Case c = make_random_vq2(M, K);
    std::vector<float> X(static_cast<size_t>(K) * N);
    for (int n = 0; n < N; ++n) {
        const std::vector<float> col = random_vec(K, 100 + n);
        std::memcpy(X.data() + static_cast<size_t>(n) * K, col.data(), K * sizeof(float));
    }
    std::vector<float> Y(static_cast<size_t>(M) * N);
    tinyqwen::matmul_vq2(c.blob.data(), X.data(), Y.data(), M, K, N);
    double e = 0.0;
    for (int n = 0; n < N; ++n) {
        std::vector<float> xcol(X.begin() + static_cast<size_t>(n) * K,
                                X.begin() + static_cast<size_t>(n + 1) * K);
        const std::vector<float> r = naive_matvec_vq2(c.blob, xcol, M, K);
        for (int m = 0; m < M; ++m) {
            e = std::max(e, std::fabs((double) Y[static_cast<size_t>(n) * M + m] - r[m]));
        }
    }
    EXPECT_TRUE(e < 1e-3);
}
