#include "test_framework.h"

#include <cstring>
#include <vector>

#include "dispatch.h"
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

TEST (matvec_dispatch_selects_impl) {
    // 按名字选择；未知名报错且不改变当前选择；用完恢复，避免影响其他测试。
    EXPECT_TRUE(set_matvec_impl_by_name("double_2_float"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "double_2_float") == 0);
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "ref") == 0);

    EXPECT_TRUE(!set_matvec_impl_by_name("no_such_impl"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "ref") == 0); // 失败不改变现状

    // 已注册列表包含两个实现（报错信息的数据源）。
    const char *avail = available_matvec_impls();
    EXPECT_TRUE(std::strstr(avail, "ref") != nullptr);
    EXPECT_TRUE(std::strstr(avail, "double_2_float") != nullptr);
}

TEST (matvec_double_2_float_matches_ref) {
    // 正确性门禁：变体走完整 dispatch 路径，输出与 ref 在容差内对齐。
    // in_dim 故意取大（257）：累加链够长，float 累加的舍入差异才显形。
    const int out_dim = 16, in_dim = 257;
    std::vector<float> w(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    std::vector<float> y_ref(out_dim), y_var(out_dim);
    matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);

    // set 必须成功——否则 matvec_f32 会静默兜底到 ref，测试变成假通过。
    EXPECT_TRUE(set_matvec_impl_by_name("double_2_float"));
    matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));

    // float 累加误差上界 ≈ in_dim * eps * max|部分和|，本组数据 ≤ ~4e-3。
    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}
