// ============================================================================
// test_topk_softmax.cpp — MoE 路由门 top-k + softmax 单元测试
// ============================================================================
// 验证 topk_softmax_ref：选 k 个最大值（降序）+ 对它们 softmax 归一（和为 1）。
//
// 测试侧独立用 numpy 风格的朴素实现对照：partial_sort 选 k、softmax 减最大值。
// ============================================================================

#include "test_framework.h"

#include "ref_ops.h"   // topk_softmax_ref

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

// 独立参考：选 top-k（降序，稳定靠前）+ softmax
void naive_topk_softmax(const std::vector<float> &logits, int k,
                        std::vector<int> &idx, std::vector<float> &w) {
    const int n = static_cast<int>(logits.size());
    const int kk = std::min(k, n);
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(kk);
    float m = logits[idx[0]];
    for (int i = 1; i < kk; ++i) m = std::max(m, logits[idx[i]]);
    std::vector<double> e(kk);
    double sum = 0.0;
    for (int i = 0; i < kk; ++i) {
        e[i] = std::exp(static_cast<double>(logits[idx[i]]) - m);
        sum += e[i];
    }
    w.resize(kk);
    for (int i = 0; i < kk; ++i) w[i] = static_cast<float>(e[i] / sum);
}

std::vector<float> random_vec(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 3.0f);
    std::vector<float> v(n);
    for (auto &x : v) x = dist(rng);
    return v;
}

} // namespace

TEST(topk_softmax_basic) {
    // 手工例：n=4, k=2, logits=[1,3,2,0] → top2 = [1(3),2(2)]
    // softmax({3,2}): exp(0)+exp(-1) = 1+0.3679 → [0.731, 0.269]
    float logits[] = {1.0f, 3.0f, 2.0f, 0.0f};
    int idx[2];
    float w[2];
    tinyqwen::topk_softmax_ref(logits, 4, 2, idx, w);
    EXPECT_EQ(idx[0], 1);
    EXPECT_EQ(idx[1], 2);
    EXPECT_NEAR(w[0], 0.7310585f, 1e-5);
    EXPECT_NEAR(w[1], 0.2689414f, 1e-5);
    // 权重和为 1
    EXPECT_NEAR(w[0] + w[1], 1.0f, 1e-5);
}

TEST(topk_softmax_vs_naive) {
    for (int n : {8, 16, 32}) {
        for (int k : {2, 4, 8}) {
            if (k > n) continue;
            const auto logits = random_vec(n, 100u + n * 7 + k);
            std::vector<int> idx(k), nidx;
            std::vector<float> w(k), nw;
            tinyqwen::topk_softmax_ref(logits.data(), n, k, idx.data(), w.data());
            naive_topk_softmax(logits, k, nidx, nw);
            // 下标集合应一致（顺序也一致——同排序谓词）
            for (int i = 0; i < k; ++i) {
                EXPECT_EQ(idx[i], nidx[i]);
                EXPECT_NEAR(w[i], nw[i], 1e-5);
            }
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) sum += w[i];
            EXPECT_NEAR(sum, 1.0f, 1e-5);
        }
    }
}

TEST(topk_softmax_descending) {
    // indices 应按 logits 降序
    const auto logits = random_vec(16, 7);
    int idx[4];
    float w[4];
    tinyqwen::topk_softmax_ref(logits.data(), 16, 4, idx, w);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(logits[idx[i]] >= logits[idx[i + 1]]);
    }
}
