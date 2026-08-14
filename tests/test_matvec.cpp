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

    // 已注册列表包含这些实现（报错信息的数据源）。
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

TEST (matvec_acc4_matches_ref) {
    // 归因阶梯之"累加结构"层：纯标量，任何平台都应存在。
    // in_dim = 4103（= 4*1025 + 3）：主循环 4 链 + 3 个标量尾段全覆盖。
    EXPECT_TRUE(set_matvec_impl_by_name("acc4"));
    const int out_dim = 16, in_dim = 4103;
    std::vector<float> w(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    std::vector<float> y_ref(out_dim), y_var(out_dim);
    matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
    matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));

    // 4 链合并顺序与 ref 串行不同，舍入差异由容差处理。
    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}

TEST (matvec_neon_nofma_matches_ref) {
    // 归因阶梯之"SIMD 宽度"层：仅 aarch64 构建注册；其他平台 skip。
    if (!set_matvec_impl_by_name("neon_nofma")) {
        std::printf("[skip] current build has no 'neon_nofma' matvec (not aarch64)\n");
        return;
    }
    // in_dim = 4103：16 主循环 / 4 向量尾段 / 3 标量尾段三段全覆盖。
    const int out_dim = 16, in_dim = 4103;
    std::vector<float> w(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    std::vector<float> y_ref(out_dim), y_var(out_dim);
    matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
    matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));

    // vmul+vadd 两次舍入，误差比 neon（FMA 一次舍入）略大，5e-3 仍宽裕。
    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}

TEST (matvec_neon_matches_ref) {
    // 正确性门禁：与 double_2_float 同款。neon 只在 aarch64 构建注册；
    // 其他平台 set 失败 -> 打印 skip 并返回，不算失败（不是静默兜底假通过，
    // 因为这里先确认了实现真的存在才往下比）。
    if (!set_matvec_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' matvec (not aarch64)\n");
        return;
    }
    // in_dim 故意取 4103 = 256*16 + 7：既不是 16 也不是 4 的倍数，
    // 主循环尾段、向量尾段、标量尾段一次全覆盖；累加链也够长，
    // float 舍入差异能显形。
    const int out_dim = 16, in_dim = 4103;
    std::vector<float> w(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    std::vector<float> y_ref(out_dim), y_var(out_dim);
    matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);

    // neon 已在上面 set 成功，这里走完整 dispatch 路径。
    matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试

    // NEON 版 4 链并行累加，每条链仅 in_dim/16 项，误差上界同数量级，5e-3 充分宽裕。
    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}

TEST (matvec_neon_mt_matches_ref) {
    // 正确性门禁：neon_mt 只在 aarch64 构建注册；其他平台 skip。
    if (!set_matvec_impl_by_name("neon_mt")) {
        std::printf("[skip] current build has no 'neon_mt' matvec (not aarch64)\n");
        return;
    }

    // 1) 并行路径：out_dim=301 故意不整除常见线程数（5/6），行块会有 ±1 的
    //    不均分；in_dim=4103 三种尾段全覆盖。循环多遍，压线程池的重复
    //    发布/归位（代数计数器、空块、参数可见性都要经得起反复）。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 2) 内联路径：权重 0.25MB < 1MB 阈值，应走单线程直算，数值同样要对。
    {
        const int out_dim = 16, in_dim = 4103;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    // 3) 极端：行数（3）少于线程数，多数线程分到空块——仍要正确归位。
    //    in_dim 取大让总权重过并行阈值；数值调小（0.0625）把长累加链的
    //    float 舍入压进容差。
    {
        const int out_dim = 3, in_dim = 100003;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.0625f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.0625f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_neon_mt_bal_matches_ref) {
    // 正确性门禁：neon_mt_bal 只在 aarch64 构建注册；其他平台 skip。
    // 加权分块改变的是"行归谁算"，行内算法与 neon_mt 逐位一致，所以
    // 覆盖重点在：不均分（取整边界）、校准收敛过程（多 job 连跑）、
    // 行数少于线程数（空块 + 首轮均分兜底）。
    if (!set_matvec_impl_by_name("neon_mt_bal")) {
        std::printf("[skip] current build has no 'neon_mt_bal' matvec (not aarch64)\n");
        return;
    }

    // 1) 并行路径 + 多 job 连跑：out_dim=301 不整除常见线程数，加权
    //    前缀和取整会有 ±1 行的段；连跑 12 遍压"校准 EMA 逐步收敛"的
    //    全过程（首 job 均分 -> 测速 -> 加权），每一步都要数值正确。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 12; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 2) 不同 shape 交替（校准权重跨 shape 复用，边界计算要对每种
    //    out_dim 都切得正确）：一个大 shape + 一个中等 shape 交错。
    {
        const int in_dim = 4103;
        std::vector<float> x(in_dim);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        for (const int out_dim : {1000, 257, 1000, 257}) {
            std::vector<float> w(static_cast<size_t>(out_dim) * in_dim);
            for (size_t i = 0; i < w.size(); ++i) {
                w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
            }
            std::vector<float> y_ref(out_dim), y_var(out_dim);
            matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 3) 内联路径：权重 0.25MB < 1MB 阈值，单线程直算。
    {
        const int out_dim = 16, in_dim = 4103;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    // 4) 极端：行数（3）少于线程数——多数线程空块、耗时 0，首轮校准的
    //    "空块用最小测量兜底"分支与空块跳过更新分支都要经得起。
    {
        const int out_dim = 3, in_dim = 100003;
        std::vector<float> w(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.0625f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.0625f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 4; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}
