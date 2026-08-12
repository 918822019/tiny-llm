#include "test_framework.h"

#include <cmath>

#include "ref_ops.h"

using namespace tinyqwen;

TEST (softmax_known_values) {
    const float x[3] = {1, 2, 3};
    float y[3];
    softmax_ref(x, y, 3);
    EXPECT_NEAR(y[0], 0.0900306, 1e-6);
    EXPECT_NEAR(y[1], 0.2447285, 1e-6);
    EXPECT_NEAR(y[2], 0.6652409, 1e-6);
    EXPECT_NEAR(y[0] + y[1] + y[2], 1.0, 1e-6);
}

TEST (softmax_stability_with_large_offset) {
    // 所有输入加同一个常数，输出不应改变（同时验证大数不上溢）。
    const float a[2] = {1000.0f, 1001.0f};
    const float b[2] = {0.0f, 1.0f};
    float ya[2], yb[2];
    softmax_ref(a, ya, 2);
    softmax_ref(b, yb, 2);
    const double e = std::exp(1.0);
    EXPECT_NEAR(ya[0], 1.0 / (1.0 + e), 1e-6);
    EXPECT_NEAR(ya[1], e / (1.0 + e), 1e-6);
    EXPECT_NEAR(ya[0], yb[0], 1e-6);
    EXPECT_NEAR(ya[1], yb[1], 1e-6);
}

TEST (silu_known_values) {
    const float x[3] = {0.0f, 1.0f, -1.0f};
    float y[3];
    silu_ref(x, y, 3);
    EXPECT_NEAR(y[0], 0.0, 1e-6);
    EXPECT_NEAR(y[1], 0.7310586, 1e-6); // sigmoid(1)
    EXPECT_NEAR(y[2], -0.2689414, 1e-6); // -1 * sigmoid(-1)
}

TEST (argmax_basic) {
    const float logits[5] = {0.1f, 3.0f, -2.0f, 3.0f, 1.5f};
    EXPECT_EQ(argmax_ref(logits, 5), 1); // 平局取第一个最大值
    const float single[1] = {42.0f};
    EXPECT_EQ(argmax_ref(single, 1), 0);
}
