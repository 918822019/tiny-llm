#include "test_framework.h"

#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

TEST (matvec_small) {
    // W = [[1,2,3],[4,5,6]]（out=2, in=3），x = [1, 0.5, -1]
    const float w[6] = {1, 2, 3, 4, 5, 6};
    const float x[3] = {1.0f, 0.5f, -1.0f};
    float y[2];
    matvec_f32_ref(w, x, y, 2, 3);
    EXPECT_NEAR(y[0], -1.0, 1e-6); // 1 + 1 - 3
    EXPECT_NEAR(y[1], 0.5, 1e-6); // 4 + 2.5 - 6
}

TEST (matvec_identity) {
    const float w[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const float x[3] = {0.25f, -3.0f, 12.5f};
    float y[3];
    matvec_f32_ref(w, x, y, 3, 3);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(y[i], x[i], 1e-6);
}

TEST (matvec_matches_naive_accumulation) {
    // 用确定性伪随机数据，和一个独立的双重循环对照。
    const int out_dim = 7, in_dim = 11;
    std::vector<float> w(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.5f;

    std::vector<float> y(out_dim);
    matvec_f32_ref(w.data(), x.data(), y.data(), out_dim, in_dim);

    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        for (int i = 0; i < in_dim; ++i) acc += (double) w[o * in_dim + i] * x[i];
        EXPECT_NEAR(y[o], acc, 1e-5);
    }
}
