#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

TEST (rmsnorm_small) {
    const float x[3] = {1.0f, 2.0f, 3.0f};
    const float w[3] = {1.0f, 1.0f, 1.0f};
    float y[3];
    rmsnorm_ref(x, w, y, 3, 1e-6f);

    // mean(x^2) = 14/3；y = x / sqrt(14/3 + 1e-6)
    const double scale = 1.0 / std::sqrt(14.0 / 3.0 + 1e-6);
    EXPECT_NEAR(y[0], 1.0 * scale, 1e-5);
    EXPECT_NEAR(y[1], 2.0 * scale, 1e-5);
    EXPECT_NEAR(y[2], 3.0 * scale, 1e-5);
}

TEST (rmsnorm_weight_and_scale) {
    // 缩放 x 不应改变归一化后的方向（RMSNorm 是尺度不变的）。
    const float w[4] = {1.0f, 2.0f, 0.5f, 4.0f};
    float y1[4], y2[4];
    const float x1[4] = {0.5f, -1.5f, 2.0f, 0.25f};
    float x2[4];
    for (int i = 0; i < 4; ++i) x2[i] = x1[i] * 7.0f;

    rmsnorm_ref(x1, w, y1, 4, 1e-6f);
    rmsnorm_ref(x2, w, y2, 4, 1e-6f);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(y1[i], y2[i], 1e-4);
}

TEST (rmsnorm_manual_value) {
    const float x[2] = {3.0f, 4.0f};
    const float w[2] = {2.0f, 1.0f};
    float y[2];
    rmsnorm_ref(x, w, y, 2, 0.0f);
    // rms = sqrt((9+16)/2) = sqrt(12.5)；y0 = 3/sqrt(12.5)*2，y1 = 4/sqrt(12.5)
    EXPECT_NEAR(y[0], 6.0 / std::sqrt(12.5), 1e-5);
    EXPECT_NEAR(y[1], 4.0 / std::sqrt(12.5), 1e-5);
}
