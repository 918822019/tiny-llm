// ============================================================================
// test_biip_rotate.cpp — BiIP 激活旋转正确性测试
// ============================================================================
// 验证 biip_rotate_activation 与"朴素归一化 Hadamard"逐字一致：
//   kronq 的正变换约定 y = blockHadamard(x) 即 每块 (x ⊙ sign / scale) @ Hᵀ，
//   H 为归一化 Sylvester Walsh-Hadamard（H/sqrt(B)）。
// 用独立实现的朴素矩阵乘法做参照，隔离测试快速 butterfly 的正确性。
// ============================================================================

#include "test_framework.h"
#include "dispatch.h"   // biip_rotate_activation

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

// 朴素 Sylvester Hadamard（归一化），大小必须是 2 的幂
std::vector<float> naive_hadamard(int n) {
    std::vector<float> H(n * n, 0.0f);
    H[0] = 1.0f;
    int size = 1;
    while (size < n) {
        for (int r = 0; r < size; ++r) {
            for (int c = 0; c < size; ++c) {
                const float v = H[r * n + c];
                H[r * n + (c + size)] = v;              // 右上 = H
                H[(r + size) * n + c] = v;              // 左下 = H
                H[(r + size) * n + (c + size)] = -v;    // 右下 = -H
            }
        }
        size *= 2;
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(n));
    for (auto &v : H) v *= inv;
    return H;
}

// 朴素参照：y = 每块 ( (x/scale) ⊙ sign ) @ Hᵀ
std::vector<float> naive_biip(const std::vector<float> &x, const float *scale,
                              const std::vector<float> &sign, int block_size) {
    const int dim = static_cast<int>(x.size());
    const int num_blocks = dim / block_size;
    const std::vector<float> H = naive_hadamard(block_size);
    std::vector<float> y(dim);
    for (int b = 0; b < num_blocks; ++b) {
        for (int r = 0; r < block_size; ++r) {          // 输出维
            float acc = 0.0f;
            for (int c = 0; c < block_size; ++c) {      // 输入维
                int gi = b * block_size + c;
                float v = x[gi];
                if (scale) v /= scale[gi];
                v *= sign[gi];
                // H 对称，@Hᵀ 即 @H；这里按 H[r][c] 累加
                acc += v * H[r * block_size + c];
            }
            y[b * block_size + r] = acc;
        }
    }
    return y;
}

std::vector<float> rand_vec(int n, unsigned seed, float lo, float hi) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v) x = d(rng);
    return v;
}

void check_biip(int dim, int block_size, bool with_scale) {
    const std::vector<float> x = rand_vec(dim, 1, -1.0f, 1.0f);
    std::vector<float> sign(dim);
    {
        std::mt19937 rng(2);
        std::uniform_int_distribution<int> b(0, 1);
        for (auto &s : sign) s = b(rng) ? 1.0f : -1.0f;
    }
    std::vector<float> scale;
    const float *scale_ptr = nullptr;
    if (with_scale) {
        scale = rand_vec(dim, 3, 0.5f, 2.0f);
        scale_ptr = scale.data();
    }
    std::vector<float> y(dim);
    tinyqwen::biip_rotate_activation(x.data(), y.data(), dim, scale_ptr, sign.data(), block_size);
    const std::vector<float> ref = naive_biip(x, scale_ptr, sign, block_size);
    double max_err = 0.0;
    for (int i = 0; i < dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    if (max_err >= 1e-4) {
        TQ_FAIL("biip_rotate dim=" + std::to_string(dim) + " bs=" + std::to_string(block_size) +
                " scale=" + std::to_string(with_scale) + " max_err=" + std::to_string(max_err));
    }
}

} // namespace

TEST(biip_rotate_bs128_with_scale)   { check_biip(896, 128, true); }
TEST(biip_rotate_bs128_no_scale)     { check_biip(896, 128, false); }
TEST(biip_rotate_bs256_with_scale)   { check_biip(4864, 256, true); }
TEST(biip_rotate_bs64)               { check_biip(256, 64, true); }
TEST(biip_rotate_single_block)       { check_biip(128, 128, true); }

// 自抵消性质：对同一向量连续施加两次"正变换"不回到原点（正变换不自逆，
// 因为 sign 在 matmul 之前）；但 biip_rotate(配对逆) 与 kronq 权重正变换配对后
// 输出不变。这里仅验证变换本身的可重复性与数值稳定。
TEST(biip_rotate_inplace_safe) {
    const int dim = 256, bs = 64;
    const std::vector<float> x = rand_vec(dim, 5, -1.0f, 1.0f);
    std::vector<float> sign(dim, 1.0f);
    std::vector<float> y1(dim), y2(dim);
    tinyqwen::biip_rotate_activation(x.data(), y1.data(), dim, nullptr, sign.data(), bs);
    // 原地：x 复制到 y2 后 x==y2 同址
    y2 = x;
    tinyqwen::biip_rotate_activation(y2.data(), y2.data(), dim, nullptr, sign.data(), bs);
    double e = 0.0;
    for (int i = 0; i < dim; ++i) e = std::max(e, std::fabs((double) y1[i] - y2[i]));
    EXPECT_TRUE(e < 1e-5);
}
