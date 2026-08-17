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

TEST (matvec_neon_mt_kv_matches_ref) {
    // 正确性门禁：neon_mt_kv 只在 aarch64 构建注册；其他平台 skip。
    // 覆盖：单矩阵并行/内联路径（与 neon_mt 同款）+ pair 并行/内联/极端形。
    if (!set_matvec_impl_by_name("neon_mt_kv")) {
        std::printf("[skip] current build has no 'neon_mt_kv' matvec (not aarch64)\n");
        return;
    }
    auto fill = [](std::vector<float> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>((i * 37 % 29) - 14) * scale;
        }
    };

    // 1) 单矩阵并行路径：301×4103 不整除线程数，连压 8 遍。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w, 0.125f);
        fill(x, 0.125f);
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 2) pair 并行路径：两矩阵各 301×4103（总量远超 0.5MB 阈值），
    //    行映射（前 out_dim 行属 w1、后半属 w2）必须切对。连压 8 遍。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<float> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w1, 0.125f);
        fill(w2, 0.0625f);
        fill(x, 0.125f);
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                            out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(y1[o], r1[o], 5e-3);
                EXPECT_NEAR(y2[o], r2[o], 5e-3);
            }
        }
    }

    // 3) pair 内联路径：各 16×2053，总量 0.25MB < 0.5MB 阈值，走两段内联。
    {
        const int out_dim = 16, in_dim = 2053;
        std::vector<float> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w1, 0.125f);
        fill(w2, 0.0625f);
        fill(x, 0.125f);
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                        out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) {
            EXPECT_NEAR(y1[o], r1[o], 5e-3);
            EXPECT_NEAR(y2[o], r2[o], 5e-3);
        }
    }

    // 4) 极端：pair 总行数 6（各 3 行）< 线程数，多数线程空块——in_dim
    //    取大让总量过 pair 并行阈值，行映射与空块归位都要对。
    {
        const int out_dim = 3, in_dim = 100003;
        std::vector<float> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w1, 0.0625f);
        fill(w2, 0.0625f);
        fill(x, 0.0625f);
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        for (int rep = 0; rep < 4; ++rep) {
            matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                            out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(y1[o], r1[o], 5e-3);
                EXPECT_NEAR(y2[o], r2[o], 5e-3);
            }
        }
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_neon_mt_kv_nt_matches_ref) {
    // 正确性门禁：neon_mt_kv_nt 只在 aarch64 构建注册；其他平台 skip。
    // LDNP 主路径要求行 16B 对齐——对齐 shape 走 ldnp，非对齐 shape 走
    // 兜底普通加载，两条路径都要覆盖。
    if (!set_matvec_impl_by_name("neon_mt_kv_nt")) {
        std::printf("[skip] current build has no 'neon_mt_kv_nt' matvec (not aarch64)\n");
        return;
    }
    auto fill = [](std::vector<float> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>((i * 37 % 29) - 14) * scale;
        }
    };

    // 1) 对齐 + 并行（LDNP 主路径）：in_dim=4104 = 32×128+8，主循环 32 批
    //    + 8 个向量尾段；行步长 4104×4 % 16 == 0，逐行对齐成立。
    {
        const int out_dim = 301, in_dim = 4104;
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w, 0.125f);
        fill(x, 0.125f);
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 2) 非对齐 + 并行（兜底路径）：in_dim=4103，第 1 行起行起点 %16 != 0。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w, 0.125f);
        fill(x, 0.125f);
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 2; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 3) 对齐 pair 并行：行映射 + LDNP 同时压。
    {
        const int out_dim = 301, in_dim = 4104;
        std::vector<float> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w1, 0.125f);
        fill(w2, 0.0625f);
        fill(x, 0.125f);
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        for (int rep = 0; rep < 4; ++rep) {
            matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                            out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(y1[o], r1[o], 5e-3);
                EXPECT_NEAR(y2[o], r2[o], 5e-3);
            }
        }
    }

    // 4) 对齐内联路径：16×4104 远低于单矩阵阈值。
    {
        const int out_dim = 16, in_dim = 4104;
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w, 0.125f);
        fill(x, 0.125f);
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    // 5) 极端：pair 总行数 6 < 线程数；in_dim=100000（对齐、n32 尾段 32），
    //    数值调小压长链舍入。
    {
        const int out_dim = 3, in_dim = 100000;
        std::vector<float> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        fill(w1, 0.0625f);
        fill(w2, 0.0625f);
        fill(x, 0.0625f);
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                        out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) {
            EXPECT_NEAR(y1[o], r1[o], 5e-3);
            EXPECT_NEAR(y2[o], r2[o], 5e-3);
        }
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (half_conversion_roundtrip) {
    // half_to_float / float_to_half 的穷举门禁：全部 65536 个 half 位形
    // （除 NaN）经 float 往返后必须逐位还原——这是 exporter 量化与 C++
    // 端一致性的地基（numpy astype("float16") 同为 RNE）。
    int checked = 0;
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint16_t h = static_cast<uint16_t>(bits);
        if ((bits & 0x7C00) == 0x7C00 && (bits & 0x3FF) != 0) continue; // NaN 不往返
        const float v = half_to_float(h);
        const uint16_t r = float_to_half(v);
        EXPECT_TRUE(half_to_float(r) == v);
        ++checked;
    }
    EXPECT_EQ(checked, 65536 - 2046); // 全部非 NaN 位形（NaN = 2 符号 × 1023 尾数）
    // 几个已知编码（手工核对过的锚点）。
    EXPECT_TRUE(float_to_half(1.0f) == 0x3C00);
    EXPECT_TRUE(float_to_half(-2.0f) == 0xC000);
    EXPECT_TRUE(float_to_half(0.5f) == 0x3800);
    EXPECT_TRUE(float_to_half(65504.0f) == 0x7BFF); // 最大正规数
    EXPECT_TRUE(float_to_half(65536.0f) == 0x7C00); // 溢出 -> inf
    EXPECT_NEAR(half_to_float(0x0400), 6.103515625e-05, 1e-12); // 最小正规数 2^-14
    EXPECT_NEAR(half_to_float(0x0001), 5.960464477539063e-08, 1e-15); // 最小非规格化
}

TEST (matvec_f16_dispatch_selects_impl) {
    // f16 注册表独立选择：set 失败不改变现状；未知实现名不兜底。
    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref"));
    EXPECT_TRUE(std::strcmp(matvec_f16_impl_name(), "ref") == 0);
    EXPECT_TRUE(!set_matvec_f16_impl_by_name("no_such_f16_impl"));
    EXPECT_TRUE(std::strcmp(matvec_f16_impl_name(), "ref") == 0);
    const char *avail = available_matvec_f16_impls();
    EXPECT_TRUE(std::strstr(avail, "ref") != nullptr);
}

TEST (matvec_f16_ref_matches_naive) {
    // f16 族参考实现的门禁：与独立的"量化后 double 累加"逐位级一致。
    const int out_dim = 7, in_dim = 11;
    std::vector<uint16_t> w(out_dim * in_dim);
    std::vector<float> x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = float_to_half(static_cast<float>((i * 37 % 29) - 14) * 0.125f);
    }
    for (int i = 0; i < in_dim; ++i) {
        x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.5f;
    }

    std::vector<float> y(out_dim);
    matvec_f16_ref(w.data(), x.data(), y.data(), out_dim, in_dim);

    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;
        for (int i = 0; i < in_dim; ++i) {
            acc += (double) half_to_float(w[o * in_dim + i]) * x[i];
        }
        EXPECT_NEAR(y[o], acc, 1e-6);
    }
}

TEST (matvec_f16_neon_mt_kv_nt_matches_ref) {
    // 正确性门禁：f16 满栈变体只在 aarch64 构建注册；其他平台 skip。
    // 覆盖与 f32 版同款：对齐 LDNP 主路径 / 非对齐兜底 / pair 行映射 /
    // 内联 / 空块极端形。权重取 0.125 的整数倍（half 可精确表示），
    // 变体与 f16_ref 的差只来自 float 累加顺序，容差沿用 5e-3。
    if (!set_matvec_f16_impl_by_name("neon_mt_kv_nt")) {
        std::printf("[skip] current build has no f16 'neon_mt_kv_nt' matvec\n");
        return;
    }
    auto fill = [](std::vector<uint16_t> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = float_to_half(static_cast<float>((i * 37 % 29) - 14) * scale);
        }
    };

    // 1) 对齐 + 并行（in_dim=4104：行步长 4104×2 % 16 == 0）。
    {
        const int out_dim = 301, in_dim = 4104;
        std::vector<uint16_t> w(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        fill(w, 0.125f);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f16_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_f16(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 2) 非对齐（in_dim=4103：第 1 行起行起点 %16 != 0，走兜底路径）。
    {
        const int out_dim = 301, in_dim = 4103;
        std::vector<uint16_t> w(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        fill(w, 0.125f);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f16_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < 2; ++rep) {
            matvec_f16(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    }

    // 3) 对齐 pair 并行：行映射 + 融合 fork-join 一起压。
    {
        const int out_dim = 301, in_dim = 4104;
        std::vector<uint16_t> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        fill(w1, 0.125f);
        fill(w2, 0.0625f);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f16_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f16_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        for (int rep = 0; rep < 4; ++rep) {
            matvec_pair_f16(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                            out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(y1[o], r1[o], 5e-3);
                EXPECT_NEAR(y2[o], r2[o], 5e-3);
            }
        }
    }

    // 4) 内联路径：16×4104 远低于单矩阵阈值。
    {
        const int out_dim = 16, in_dim = 4104;
        std::vector<uint16_t> w(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        fill(w, 0.125f);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f16_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f16(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    // 5) 极端：pair 总行数 6 < 线程数；in_dim=100000（对齐、尾段 32）。
    {
        const int out_dim = 3, in_dim = 100000;
        std::vector<uint16_t> w1(static_cast<size_t>(out_dim) * in_dim),
                w2(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        fill(w1, 0.0625f);
        fill(w2, 0.0625f);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.0625f;
        std::vector<float> r1(out_dim), r2(out_dim), y1(out_dim), y2(out_dim);
        matvec_f16_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f16_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        matvec_pair_f16(w1.data(), w2.data(), x.data(), y1.data(), y2.data(),
                        out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) {
            EXPECT_NEAR(y1[o], r1[o], 5e-3);
            EXPECT_NEAR(y2[o], r2[o], 5e-3);
        }
    }

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_pair_fallback_matches_ref) {
    // 中性门禁：未注册 pair 的 impl，matvec_pair_f32 必须等价于分开调两次
    // matvec_f32——runtime 改成 pair 调用后，ref 等 impl 的数值行为不变。
    const int out_dim = 19, in_dim = 137;
    std::vector<float> w1(out_dim * in_dim), w2(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w1[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        w2[i] = static_cast<float>((i * 23 % 31) - 15) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    // ref：兜底路径（ref 不注册 pair）。
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));
    std::vector<float> y1(out_dim), y2(out_dim), r1(out_dim), r2(out_dim);
    matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(), out_dim, in_dim);
    matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
    matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
    for (int o = 0; o < out_dim; ++o) {
        EXPECT_NEAR(y1[o], r1[o], 1e-6); // 同一条代码路径，要求逐位级一致
        EXPECT_NEAR(y2[o], r2[o], 1e-6);
    }

    // double_2_float：同样未注册 pair，兜底后数值应与"分开调该 impl"一致。
    EXPECT_TRUE(set_matvec_impl_by_name("double_2_float"));
    matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(), out_dim, in_dim);
    matvec_f32(w1.data(), x.data(), r1.data(), out_dim, in_dim);
    matvec_f32(w2.data(), x.data(), r2.data(), out_dim, in_dim);
    for (int o = 0; o < out_dim; ++o) {
        EXPECT_NEAR(y1[o], r1[o], 1e-6);
        EXPECT_NEAR(y2[o], r2[o], 1e-6);
    }
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));
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

TEST (matvec_cuda_matches_ref) {
    // 正确性门禁：cuda 只在检测到 nvcc 的构建里注册（见
    // kernels/CMakeLists.txt 的 check_language(CUDA)）；没有 CUDA 工具链的
    // 平台（mac/Android/普通 CI）set 失败 -> skip，与 neon 系列在 x86 上
    // 的行为对称，不是静默兜底假通过。
    if (!set_matvec_impl_by_name("cuda")) {
        std::printf("[skip] current build has no 'cuda' matvec (no CUDA toolchain)\n");
        return;
    }

    // 1) 常规形状：out_dim/in_dim 都不是任何 2 的幂对齐边界，和其他变体
    //    测试同款输入构造方式，误差量级参照 neon 系列（float 累加，5e-3）。
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

    // 2) out_dim 远大于一个 block 的 thread 数（覆盖 kernel 里的
    //    grid-stride loop：block 数量封顶后仍要循环覆盖所有行），
    //    in_dim 较小（模拟 k/v_proj 这类小矩阵形状）。
    {
        const int out_dim = 2000, in_dim = 67;
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_cuda_resident_matches_ref) {
    // 正确性门禁：cuda_resident 只在 CUDA 构建注册；其他平台 skip。
    // 权重常驻缓存的正确性重点：
    //   1. 首次调用（冷路径：上传 + 登记）结果正确；
    //   2. 重复调用（热路径：命中缓存复用 device 副本）结果仍正确——
    //      连跑多遍，确保缓存命中分支没有读到脏/错位数据；
    //   3. 两块不同 host 指针的同形状权重交替调用——缓存按指针区分，
    //      不能串线（A 的调用拿到 B 的 device 副本）。
    if (!set_matvec_impl_by_name("cuda_resident")) {
        std::printf("[skip] current build has no 'cuda_resident' matvec (no CUDA toolchain)\n");
        return;
    }

    // 1+2) 同一块权重连跑 12 遍：第 1 遍走冷路径，其后全走缓存命中。
    {
        const int out_dim = 16, in_dim = 4103;
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

    // 3) 两块不同指针的同形状权重交替调用：缓存按 host 指针区分，
    //    交替 8 轮，任何一轮结果错误都说明缓存串线。
    {
        const int out_dim = 8, in_dim = 501;
        std::vector<float> wa(out_dim * in_dim), wb(out_dim * in_dim), x(in_dim);
        for (int i = 0; i < out_dim * in_dim; ++i) {
            wa[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
            wb[i] = static_cast<float>((i * 41 % 23) - 11) * 0.125f; // 不同的值
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

        std::vector<float> ya(out_dim), yb(out_dim), ra(out_dim), rb(out_dim);
        matvec_f32_ref(wa.data(), x.data(), ra.data(), out_dim, in_dim);
        matvec_f32_ref(wb.data(), x.data(), rb.data(), out_dim, in_dim);
        for (int rep = 0; rep < 8; ++rep) {
            matvec_f32(wa.data(), x.data(), ya.data(), out_dim, in_dim);
            matvec_f32(wb.data(), x.data(), yb.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(ya[o], ra[o], 5e-3);
                EXPECT_NEAR(yb[o], rb[o], 5e-3);
            }
        }
    }

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_cuda_resident_ws_matches_ref) {
    // 正确性门禁：cuda_resident_ws 只在 CUDA 构建注册；其他平台 skip。
    // 在 cuda_resident 的缓存正确性之上，额外覆盖 workspace 的 grow-only
    // 逻辑：
    //   1. 小 shape 先跑（workspace 按小容量分配）；
    //   2. 换成更大 shape（触发扩容重分配）；
    //   3. 再换回小 shape（复用大容量缓冲跑小 shape，只读写前 n 个元素，
    //      不能因为缓冲变大而读脏数据或写越界）。
    if (!set_matvec_impl_by_name("cuda_resident_ws")) {
        std::printf("[skip] current build has no 'cuda_resident_ws' matvec (no CUDA toolchain)\n");
        return;
    }

    auto check = [](const int out_dim, const int in_dim, int reps) {
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < reps; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    };

    check(16, 4103, 4);    // 1) 小 out_dim，冷启动 + 缓存命中
    check(2000, 4103, 2);  // 2) 大 out_dim，触发 d_y 扩容
    check(8, 501, 4);      // 3) 换回小 shape，复用大缓冲
    check(2000, 67, 2);    // 2') 大 out_dim + 小 in_dim（d_x 不扩容，d_y 已够）

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_cuda_resident_coal_matches_ref) {
    // 正确性门禁：cuda_resident_coal 只在 CUDA 构建注册；其他平台 skip。
    // 合并访存 kernel 的正确性重点：
    //   1. in_dim 不是 blockDim(256) 的倍数——跨步循环的尾部由低编号线程
    //      自然收尾，不能漏算/越界；
    //   2. in_dim < blockDim——高编号线程乘加循环一轮都不跑（部分和为 0），
    //      归约仍须正确（k/v_proj 类小矩阵 + 测试形状都要覆盖）；
    //   3. 大 out_dim（lm_head 形状量级）——block-per-row 一行一个 block。
    if (!set_matvec_impl_by_name("cuda_resident_coal")) {
        std::printf("[skip] current build has no 'cuda_resident_coal' matvec (no CUDA toolchain)\n");
        return;
    }

    auto check = [](const int out_dim, const int in_dim, int reps) {
        std::vector<float> w(static_cast<size_t>(out_dim) * in_dim), x(in_dim);
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < reps; ++rep) {
            matvec_f32(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    };

    check(16, 4103, 4);    // 1) in_dim % 256 != 0（4103 = 16*256 + 7）
    check(8, 67, 4);       // 2) in_dim < 256：多数线程空转，归约兜底
    check(2000, 67, 2);    // 2') 同上但大 out_dim
    check(3, 256, 4);      // in_dim 恰好一个 block 步长
    check(128, 896, 2);    // k/v_proj 真实形状

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认，避免影响其他测试
}

TEST (matvec_f16_cuda_resident_coal_matches_ref) {
    // 正确性门禁：f16 cuda_resident_coal 只在 CUDA 构建注册；其他平台 skip。
    // 对齐基准是 f16 族的 matvec_f16_ref（不是 f32 的）——权重已半精度量化。
    // 权重取 0.125 的整数倍（half 可精确表示），变体与 f16_ref 的差只来自
    // fp32 累加顺序，容差沿用 5e-3。边界覆盖同 f32 coal 版：in_dim 非 256
    // 倍数、in_dim<256（高编号线程空转）、大 out_dim block-per-row。
    if (!set_matvec_f16_impl_by_name("cuda_resident_coal")) {
        std::printf("[skip] current build has no f16 'cuda_resident_coal' matvec (no CUDA toolchain)\n");
        return;
    }

    auto check = [](const int out_dim, const int in_dim, int reps) {
        std::vector<uint16_t> w(static_cast<size_t>(out_dim) * in_dim);
        std::vector<float> x(in_dim);
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            w[i] = float_to_half(static_cast<float>((i * 37 % 29) - 14) * 0.125f);
        }
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y_ref(out_dim), y_var(out_dim);
        matvec_f16_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
        for (int rep = 0; rep < reps; ++rep) {
            matvec_f16(w.data(), x.data(), y_var.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
        }
    };

    check(16, 4103, 4);    // in_dim % 256 != 0
    check(8, 67, 4);       // in_dim < 256：多数线程空转
    check(2000, 67, 2);    // 大 out_dim + 小 in_dim
    check(3, 256, 4);      // in_dim 恰一个 block 步长
    check(128, 896, 2);    // k/v_proj 真实形状

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认
}
