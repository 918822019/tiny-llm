// ============================================================================
// test_quant_ops.cpp — 评测/量化辅助函数单元测试
// ============================================================================
// 覆盖 PPL 评测与紧凑 INT4 embed 用到的两个共享 helper：
//   - log_softmax_at：PPL 交叉熵的核心（数值稳定的 log-softmax 单点取值）
//   - dequant_i4_row：紧凑 INT4 embed 查表的行反量化
// ============================================================================

#include "test_framework.h"
#include "ref_ops.h"   // log_softmax_at / dequant_i4_row / float_to_half / half_to_float

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// 朴素 log-softmax 参照：log( exp(logits[t]) / Σ exp(logits) )
double naive_log_softmax(const std::vector<float> &logits, int target) {
    double mx = logits[0];
    for (size_t v = 1; v < logits.size(); ++v) if (logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    for (float x : logits) sum += std::exp(static_cast<double>(x) - mx);
    return (static_cast<double>(logits[target]) - mx) - std::log(sum);
}

} // namespace

// log_softmax_at 与朴素参照一致（含数值稳定：大 logits 不溢出）
TEST(log_softmax_at_matches_naive) {
    const std::vector<float> logits = {1.0f, 2.5f, -0.5f, 3.0f, 0.0f};
    for (int t = 0; t < static_cast<int>(logits.size()); ++t) {
        const double got = tinyqwen::log_softmax_at(logits.data(), logits.size(), t);
        const double ref = naive_log_softmax(logits, t);
        EXPECT_TRUE(std::fabs(got - ref) < 1e-9);
    }
}

// 数值稳定性：很大的 logits（直接 exp 会溢出）仍应给出有限且正确的结果
TEST(log_softmax_at_stable_large_logits) {
    const std::vector<float> logits = {1000.0f, 1001.0f, 999.0f};
    const double got = tinyqwen::log_softmax_at(logits.data(), logits.size(), 1);
    const double ref = naive_log_softmax(logits, 1);
    EXPECT_TRUE(std::isfinite(got));
    EXPECT_TRUE(std::fabs(got - ref) < 1e-9);
    // 最大值的 log-softmax 应接近 0（概率接近 1）
    const double got_max = tinyqwen::log_softmax_at(logits.data(), logits.size(), 1);
    EXPECT_TRUE(got_max < 0.0 && got_max > -1.0);
}

// log_softmax_at 归一化：Σ exp(log_softmax) == 1
TEST(log_softmax_at_normalizes) {
    const std::vector<float> logits = {0.3f, -1.2f, 2.0f, 0.7f};
    double sum = 0.0;
    for (int t = 0; t < static_cast<int>(logits.size()); ++t)
        sum += std::exp(tinyqwen::log_softmax_at(logits.data(), logits.size(), t));
    EXPECT_TRUE(std::fabs(sum - 1.0) < 1e-6);
}

// dequant_i4_row：构造一个 2 组的小 i4 行，逐值核对反量化
TEST(dequant_i4_row_correct) {
    const int dim = 8, gs = 4;               // 2 组，每组 4 个元素
    const int group_total = 4 + gs / 2;      // 4B 头 + 2B 数据
    std::vector<uint8_t> row(2 * group_total, 0);

    // 组 0: scale=0.5, zero=8；量化值 q=[0,8,15,3] → dequant=(q-8)*0.5
    const float scale0 = 0.5f, zero0 = 8.0f;
    uint16_t s0 = tinyqwen::float_to_half(scale0), z0 = tinyqwen::float_to_half(zero0);
    std::memcpy(row.data(), &s0, 2);
    std::memcpy(row.data() + 2, &z0, 2);
    row[4] = (8 << 4) | 0;   // 元素0=0(低), 元素1=8(高)
    row[5] = (3 << 4) | 15;  // 元素2=15(低), 元素3=3(高)

    // 组 1: scale=0.25, zero=4；q=[4,4,4,4] → dequant=(4-4)*0.25=0
    const float scale1 = 0.25f, zero1 = 4.0f;
    uint16_t s1 = tinyqwen::float_to_half(scale1), z1 = tinyqwen::float_to_half(zero1);
    std::memcpy(row.data() + group_total, &s1, 2);
    std::memcpy(row.data() + group_total + 2, &z1, 2);
    row[group_total + 4] = (4 << 4) | 4;
    row[group_total + 5] = (4 << 4) | 4;

    std::vector<float> out(dim);
    tinyqwen::dequant_i4_row(row.data(), 0, dim, gs, out.data());

    // 期望：组0 = (q-8)*0.5 = [-4, 0, 3.5, -2.5]；组1 = 0
    const float expect[8] = {-4.0f, 0.0f, 3.5f, -2.5f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < dim; ++i) {
        EXPECT_NEAR(out[i], expect[i], 1e-3);
    }
}

// dequant_i4_row：尾部不足一组（dim 非 gs 整数倍）只反量化有效元素
TEST(dequant_i4_row_partial_tail) {
    const int dim = 6, gs = 4;               // 1 满组 + 尾部 2 个
    const int groups = 2;
    const int group_total = 4 + gs / 2;
    std::vector<uint8_t> row(groups * group_total, 0);
    const float scale = 1.0f, zero = 0.0f;
    uint16_t s = tinyqwen::float_to_half(scale), z = tinyqwen::float_to_half(zero);
    for (int g = 0; g < groups; ++g) {
        std::memcpy(row.data() + g * group_total, &s, 2);
        std::memcpy(row.data() + g * group_total + 2, &z, 2);
        // 全 5：低/高 nibble 都是 5
        row[g * group_total + 4] = (5 << 4) | 5;
        row[g * group_total + 5] = (5 << 4) | 5;
    }
    std::vector<float> out(dim);
    tinyqwen::dequant_i4_row(row.data(), 0, dim, gs, out.data());
    // scale=1, zero=0, q=5 → 全部应为 5（只前 6 个有效）
    for (int i = 0; i < dim; ++i) EXPECT_NEAR(out[i], 5.0f, 1e-3);
}
