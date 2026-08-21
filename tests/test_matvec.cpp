// ============================================================================
// test_matvec.cpp — 矩阵-向量乘法（matvec）内核的全面单元测试
// ============================================================================
// 本文件是 tiny-llm 项目中最大、最重要的测试文件之一，覆盖所有 matvec
// 实现变体的正确性验证，包括：
//
//   f32 族：ref, double_2_float, acc4, neon, neon_nofma, neon_mt,
//           neon_mt_kv, neon_mt_kv_nt, neon_mt_bal, cuda, cuda_resident,
//           cuda_resident_ws, cuda_resident_coal, cuda_resident_coal_ws
//
//   f16 族：ref, neon_mt_kv_nt, cuda_resident_coal, cuda_resident_coal_ws,
//           cuda_fused (pair + qkv)
//
//   i4 族：ref, neon
//
//   辅助函数：half_conversion_roundtrip
//
// 每个测试遵循"归因阶梯"方法论：从最简单的 ref 实现开始，逐层验证更复杂
// 的变体（标量优化 → SIMD 宽度 → 多线程并行 → 加权分块 → GPU）。
// 每个变体都通过 dispatch 路径调用，并与 ref 基准对齐。
// ============================================================================

#include "test_framework.h"

#include <cstring>
#include <vector>

#include "dispatch.h"
#include "ref_ops.h"

using namespace tinyqwen;

// =========================================================================
//  f32 基础 matvec 测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：matvec 小矩阵手动计算
//
// W = [[1,2,3],[4,5,6]]（out_dim=2, in_dim=3），x = [1, 0.5, -1]
// 手动验证：
//   y[0] = 1*1 + 2*0.5 + 3*(-1) = 1 + 1 - 3 = -1
//   y[1] = 4*1 + 5*0.5 + 6*(-1) = 4 + 2.5 - 6 = 0.5
// ---------------------------------------------------------------------------
TEST (matvec_small) {
    // W = [[1,2,3],[4,5,6]]（out=2, in=3），x = [1, 0.5, -1]
    const float w[6] = {1, 2, 3, 4, 5, 6}; // 行优先存储
    const float x[3] = {1.0f, 0.5f, -1.0f};
    float y[2];

    matvec_f32_ref(w, x, y, 2, 3);

    EXPECT_NEAR(y[0], -1.0, 1e-6); // 1 + 1 - 3
    EXPECT_NEAR(y[1], 0.5, 1e-6);  // 4 + 2.5 - 6
}

// ---------------------------------------------------------------------------
// 测试：matvec 单位矩阵验证
//
// 如果 W 是单位矩阵，则 y 应该等于 x（恒等变换）。
// ---------------------------------------------------------------------------
TEST (matvec_identity) {
    const float w[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // 3×3 单位矩阵
    const float x[3] = {0.25f, -3.0f, 12.5f};
    float y[3];

    matvec_f32_ref(w, x, y, 3, 3);

    for (int i = 0; i < 3; ++i) EXPECT_NEAR(y[i], x[i], 1e-6);
}

// ---------------------------------------------------------------------------
// 测试：matvec 与独立双重循环对比
//
// 用确定性伪随机数据，对比 ref 实现与独立编写的双重循环累加结果。
// ---------------------------------------------------------------------------
TEST (matvec_matches_naive_accumulation) {
    // 用确定性伪随机数据，和一个独立的双重循环对照。
    const int out_dim = 7, in_dim = 11;
    std::vector<float> w(out_dim * in_dim), x(in_dim);

    // 确定性伪随机数据生成
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.5f;

    std::vector<float> y(out_dim);
    matvec_f32_ref(w.data(), x.data(), y.data(), out_dim, in_dim);

    // 独立双重循环验证
    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0; // 使用 double 累加以减少浮点误差
        for (int i = 0; i < in_dim; ++i) acc += (double) w[o * in_dim + i] * x[i];
        EXPECT_NEAR(y[o], acc, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// 测试：matvec dispatch 系统——按名称选择实现
//
// 验证 set_matvec_impl_by_name 的行为：
//   1. 可以切换到已注册的实现
//   2. 失败的设置（不存在实现名）不改变当前选择
//   3. available_matvec_impls() 列出了所有已注册项
// ---------------------------------------------------------------------------
TEST (matvec_dispatch_selects_impl) {
    // 按名字选择；未知名报错且不改变当前选择；用完恢复，避免影响其他测试。

    // 切换到 double_2_float 实现
    EXPECT_TRUE(set_matvec_impl_by_name("double_2_float"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "double_2_float") == 0);

    // 切换回 ref
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "ref") == 0);

    // 不存在的实现名——应该失败，且当前选择不变
    EXPECT_TRUE(!set_matvec_impl_by_name("no_such_impl"));
    EXPECT_TRUE(std::strcmp(matvec_impl_name(), "ref") == 0); // 失败不改变现状

    // 已注册列表包含这些实现
    const char *avail = available_matvec_impls();
    EXPECT_TRUE(std::strstr(avail, "ref") != nullptr);
    EXPECT_TRUE(std::strstr(avail, "double_2_float") != nullptr);
}

// =========================================================================
//  f32 变体正确性测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：double_2_float 变体与 ref 对齐
//
// 权重和 x 用 float 存储，但累加器用 double（双精度），减少累加误差。
// in_dim 故意取大（257），让 float 累加的舍入差异显形。
// ---------------------------------------------------------------------------
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
    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认

    // float 累加误差上界 ≈ in_dim * eps * max|部分和|，本组数据 ≤ ~4e-3。
    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}

// ---------------------------------------------------------------------------
// 测试：acc4 变体与 ref 对齐
//
// acc4 使用 4 条独立的累加链（减少 FMA 流水线停顿），
// 累加顺序与 ref 的串行不同，舍入误差由容差处理。
// in_dim = 4103 = 4*1025 + 3，覆盖主循环 4 链 + 3 个标量尾段。
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 测试：neon_nofma 变体与 ref 对齐
//
// 仅 aarch64 构建注册；其他平台 skip。
// neon_nofma 使用 vmul+vadd 两次舍入，误差比 FMA（一次舍入）略大。
// in_dim = 4103 覆盖 16 主循环 / 4 向量尾段 / 3 标量尾段。
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// 测试：neon 变体与 ref 对齐
//
// in_dim = 4103 = 256*16 + 7，三种尾段全覆盖。
// NEON 版 4 链并行累加，每条链仅 in_dim/16 项，容差 5e-3。
// ---------------------------------------------------------------------------
TEST (matvec_neon_matches_ref) {
    if (!set_matvec_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' matvec (not aarch64)\n");
        return;
    }
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
    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认

    for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_var[o], y_ref[o], 5e-3);
}

// ---------------------------------------------------------------------------
// 测试：neon_mt（多线程 NEON）变体与 ref 对齐
//
// 覆盖三种场景：
//   1. 并行路径：out_dim=301 不整除常见线程数，行块会有 ±1 的不均分
//   2. 内联路径：权重 0.25MB < 1MB 阈值，单线程直算
//   3. 极端：只有 3 行，多数线程分到空块，仍要正确归位
// ---------------------------------------------------------------------------
TEST (matvec_neon_mt_matches_ref) {
    if (!set_matvec_impl_by_name("neon_mt")) {
        std::printf("[skip] current build has no 'neon_mt' matvec (not aarch64)\n");
        return;
    }

    // 1) 并行路径：out_dim=301 不整除线程数，连压 8 遍
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

    // 2) 内联路径：权重 0.25MB < 1MB 阈值，单线程直算
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

    // 3) 极端：行数（3）少于线程数，多数线程空块
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：neon_mt_kv 变体与 ref 对齐
//
// 覆盖单矩阵和 pair（双矩阵）两种场景，各含并行/内联/极端形。
// pair 模式用于门控注意力层（gate/up）的融合计算。
// ---------------------------------------------------------------------------
TEST (matvec_neon_mt_kv_matches_ref) {
    if (!set_matvec_impl_by_name("neon_mt_kv")) {
        std::printf("[skip] current build has no 'neon_mt_kv' matvec (not aarch64)\n");
        return;
    }
    auto fill = [](std::vector<float> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>((i * 37 % 29) - 14) * scale;
        }
    };

    // 1) 单矩阵并行路径：301×4103 不整除线程数，连压 8 遍
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

    // 2) pair 并行路径：两矩阵各 301×4103（总量远超 0.5MB 阈值）
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

    // 3) pair 内联路径：各 16×2053，总量 0.25MB < 0.5MB 阈值
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

    // 4) 极端：pair 总行数 6（各 3 行）< 线程数，多数线程空块
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：neon_mt_kv_nt 变体与 ref 对齐
//
// 使用 LDNP（128-bit non-temporal pair load）指令优化内存访问。
// 要求行步长 16B 对齐——对齐 shape 走 ldnp，非对齐走兜底普通加载。
// 覆盖：对齐 LDNB 主路径 / 非对齐兜底 / pair 行映射 / 内联 / 极端形。
// ---------------------------------------------------------------------------
TEST (matvec_neon_mt_kv_nt_matches_ref) {
    if (!set_matvec_impl_by_name("neon_mt_kv_nt")) {
        std::printf("[skip] current build has no 'neon_mt_kv_nt' matvec (not aarch64)\n");
        return;
    }
    auto fill = [](std::vector<float> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = static_cast<float>((i * 37 % 29) - 14) * scale;
        }
    };

    // 1) 对齐 + 并行（LDNP 主路径）：in_dim=4104 = 32×128+8
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

    // 2) 非对齐 + 并行（兜底路径）：in_dim=4103
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

    // 3) 对齐 pair 并行：行映射 + LDNP 同时压
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

    // 4) 对齐内联路径：16×4104 远低于单矩阵阈值
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

    // 5) 极端：pair 总行数 6 < 线程数；in_dim=100000
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// =========================================================================
//  half 转换工具函数测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：half_to_float / float_to_half 往返穷举
//
// 遍历全部 65536 个 half 位形（除 NaN），验证经 float 往返后逐位还原。
// 这是"地基"级测试：exporter 量化与 C++ 端一致性完全依赖这两个函数。
// 采用 RNE（Round to Nearest Even）舍入模式。
// ---------------------------------------------------------------------------
TEST (half_conversion_roundtrip) {
    int checked = 0;
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint16_t h = static_cast<uint16_t>(bits);
        // NaN 不参与往返（NaN 的浮点表示不唯一）
        if ((bits & 0x7C00) == 0x7C00 && (bits & 0x3FF) != 0) continue;
        const float v = half_to_float(h);
        const uint16_t r = float_to_half(v);
        EXPECT_TRUE(half_to_float(r) == v);
        ++checked;
    }
    EXPECT_EQ(checked, 65536 - 2046); // 全部非 NaN 位形（NaN = 2 符号 × 1023 尾数）

    // 几个已知编码（手工核对过的锚点）
    EXPECT_TRUE(float_to_half(1.0f) == 0x3C00);       // 1.0
    EXPECT_TRUE(float_to_half(-2.0f) == 0xC000);      // -2.0
    EXPECT_TRUE(float_to_half(0.5f) == 0x3800);       // 0.5
    EXPECT_TRUE(float_to_half(65504.0f) == 0x7BFF);   // 最大正规数
    EXPECT_TRUE(float_to_half(65536.0f) == 0x7C00);   // 溢出 -> inf
    EXPECT_NEAR(half_to_float(0x0400), 6.103515625e-05, 1e-12);   // 最小正规数 2^-14
    EXPECT_NEAR(half_to_float(0x0001), 5.960464477539063e-08, 1e-15); // 最小非规格化
}

// =========================================================================
//  f16 族 matvec 测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：f16 matvec dispatch 系统
// ---------------------------------------------------------------------------
TEST (matvec_f16_dispatch_selects_impl) {
    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref"));
    EXPECT_TRUE(std::strcmp(matvec_f16_impl_name(), "ref") == 0);
    EXPECT_TRUE(!set_matvec_f16_impl_by_name("no_such_f16_impl"));
    EXPECT_TRUE(std::strcmp(matvec_f16_impl_name(), "ref") == 0);
    const char *avail = available_matvec_f16_impls();
    EXPECT_TRUE(std::strstr(avail, "ref") != nullptr);
}

// ---------------------------------------------------------------------------
// 测试：f16 ref 实现与独立"量化后 double 累加"对比
// ---------------------------------------------------------------------------
TEST (matvec_f16_ref_matches_naive) {
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

// ---------------------------------------------------------------------------
// 测试：f16 neon_mt_kv_nt 变体与 ref 对齐
//
// 权重取 0.125 的整数倍（half 可精确表示），
// 变体与 f16_ref 的差只来自 float 累加顺序，容差 5e-3。
// 覆盖：对齐 LDNP / 非对齐兜底 / pair 行映射 / 内联 / 极端形。
// ---------------------------------------------------------------------------
TEST (matvec_f16_neon_mt_kv_nt_matches_ref) {
    if (!set_matvec_f16_impl_by_name("neon_mt_kv_nt")) {
        std::printf("[skip] current build has no f16 'neon_mt_kv_nt' matvec\n");
        return;
    }
    auto fill = [](std::vector<uint16_t> &v, float scale) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] = float_to_half(static_cast<float>((i * 37 % 29) - 14) * scale);
        }
    };

    // 1) 对齐 + 并行（in_dim=4104：行步长 4104×2 % 16 == 0）
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

    // 2) 非对齐（in_dim=4103）
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

    // 3) 对齐 pair 并行
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

    // 4) 内联路径：16×4104
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

    // 5) 极端：pair 总行数 6 < 线程数；in_dim=100000
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

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：pair fallback 路径——未注册 pair 的实现必须等价于分开调两次 matvec
// ---------------------------------------------------------------------------
TEST (matvec_pair_fallback_matches_ref) {
    const int out_dim = 19, in_dim = 137;
    std::vector<float> w1(out_dim * in_dim), w2(out_dim * in_dim), x(in_dim);
    for (int i = 0; i < out_dim * in_dim; ++i) {
        w1[i] = static_cast<float>((i * 37 % 29) - 14) * 0.125f;
        w2[i] = static_cast<float>((i * 23 % 31) - 15) * 0.125f;
    }
    for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;

    // ref 兜底路径（ref 不注册 pair）
    EXPECT_TRUE(set_matvec_impl_by_name("ref"));
    std::vector<float> y1(out_dim), y2(out_dim), r1(out_dim), r2(out_dim);
    matvec_pair_f32(w1.data(), w2.data(), x.data(), y1.data(), y2.data(), out_dim, in_dim);
    matvec_f32_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
    matvec_f32_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
    for (int o = 0; o < out_dim; ++o) {
        EXPECT_NEAR(y1[o], r1[o], 1e-6); // 同一条代码路径，要求逐位级一致
        EXPECT_NEAR(y2[o], r2[o], 1e-6);
    }

    // double_2_float 兜底路径
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

// ---------------------------------------------------------------------------
// 测试：neon_mt_bal（加权分块）变体与 ref 对齐
//
// 加权分块改变的是"行归谁算"，行内算法与 neon_mt 逐位一致。
// 覆盖重点：不均分（取整边界）、校准收敛过程（多 job 连跑）、
// 行数少于线程数（空块 + 首轮均分兜底）、不同 shape 交替。
// ---------------------------------------------------------------------------
TEST (matvec_neon_mt_bal_matches_ref) {
    if (!set_matvec_impl_by_name("neon_mt_bal")) {
        std::printf("[skip] current build has no 'neon_mt_bal' matvec (not aarch64)\n");
        return;
    }

    // 1) 并行 + 多 job 连跑：out_dim=301，连跑 12 遍
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

    // 2) 不同 shape 交替（校准权重跨 shape 复用）
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

    // 3) 内联路径：权重 0.25MB < 1MB 阈值
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

    // 4) 极端：行数（3）少于线程数
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// =========================================================================
//  CUDA matvec 测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：cuda 变体与 ref 对齐
//
// 只在 CUDA 构建注册，其他平台 skip。
// 覆盖：常规矩阵 + 大 out_dim（grid-stride loop 覆盖）
// ---------------------------------------------------------------------------
TEST (matvec_cuda_matches_ref) {
    if (!set_matvec_impl_by_name("cuda")) {
        std::printf("[skip] current build has no 'cuda' matvec (no CUDA toolchain)\n");
        return;
    }

    // 1) 常规形状
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

    // 2) out_dim 远大于一个 block 的 thread 数（覆盖 grid-stride loop）
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：cuda_resident 变体与 ref 对齐
//
// 权重常驻 GPU 缓存：首次调用走冷路径（上传+登记），后续命中缓存。
// 覆盖连跑多遍 + 两块不同指针交替。
// ---------------------------------------------------------------------------
TEST (matvec_cuda_resident_matches_ref) {
    if (!set_matvec_impl_by_name("cuda_resident")) {
        std::printf("[skip] current build has no 'cuda_resident' matvec (no CUDA toolchain)\n");
        return;
    }

    // 1+2) 同一块权重连跑 12 遍：第 1 遍冷路径，其后热路径
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

    // 3) 两块不同指针交替调用：缓存按 host 指针区分，不能串线
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

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：cuda_resident_ws 变体与 ref 对齐
//
// 在缓存正确性之上，额外覆盖 workspace 的 grow-only 逻辑：
// 小 shape → 大 shape（触发扩容）→ 回小 shape（复用大缓冲）。
// ---------------------------------------------------------------------------
TEST (matvec_cuda_resident_ws_matches_ref) {
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

    check(16, 4103, 4);    // 小 out_dim，冷启动 + 缓存命中
    check(2000, 4103, 2);  // 大 out_dim，触发 d_y 扩容
    check(8, 501, 4);      // 换回小 shape，复用大缓冲
    check(2000, 67, 2);    // 大 out_dim + 小 in_dim

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：cuda_resident_coal（合并访存）变体与 ref 对齐
//
// 覆盖：in_dim 非 256 倍数、in_dim < 256（高编号线程空转）、大 out_dim。
// ---------------------------------------------------------------------------
TEST (matvec_cuda_resident_coal_matches_ref) {
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

    check(16, 4103, 4);    // in_dim % 256 != 0
    check(8, 67, 4);       // in_dim < 256：多数线程空转，归约兜底
    check(2000, 67, 2);    // 大 out_dim + 小 in_dim
    check(3, 256, 4);      // in_dim 恰好一个 block 步长
    check(128, 896, 2);    // k/v_proj 真实形状

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：f16 cuda_resident_coal 变体与 ref 对齐
// ---------------------------------------------------------------------------
TEST (matvec_f16_cuda_resident_coal_matches_ref) {
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
    check(8, 67, 4);       // in_dim < 256
    check(2000, 67, 2);    // 大 out_dim + 小 in_dim
    check(3, 256, 4);      // in_dim 恰一个 block 步长
    check(128, 896, 2);    // k/v_proj 真实形状

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：cuda_resident_coal_ws 变体与 ref 对齐
// ---------------------------------------------------------------------------
TEST (matvec_cuda_resident_coal_ws_matches_ref) {
    if (!set_matvec_impl_by_name("cuda_resident_coal_ws")) {
        std::printf("[skip] current build has no 'cuda_resident_coal_ws' matvec (no CUDA toolchain)\n");
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

    check(16, 4103, 4);    // 小 out_dim，冷启动 + 连跑
    check(2000, 4103, 2);  // 大 out_dim，触发 d_y 扩容
    check(8, 67, 4);       // in_dim<256 + 回小 shape 复用大缓冲
    check(2000, 67, 2);    // 大 out_dim 小 in_dim
    check(128, 896, 4);    // k/v_proj 真实形状，连跑压时序

    EXPECT_TRUE(set_matvec_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：f16 cuda_resident_coal_ws 变体与 ref 对齐
// ---------------------------------------------------------------------------
TEST (matvec_f16_cuda_resident_coal_ws_matches_ref) {
    if (!set_matvec_f16_impl_by_name("cuda_resident_coal_ws")) {
        std::printf("[skip] current build has no f16 'cuda_resident_coal_ws' matvec (no CUDA toolchain)\n");
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

    check(16, 4103, 4);
    check(2000, 4103, 2);  // 触发 d_y 扩容
    check(8, 67, 4);       // in_dim<256 + 回小 shape
    check(128, 896, 4);    // k/v_proj 形状，连跑压时序

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认
}

// ---------------------------------------------------------------------------
// 测试：f16 融合 pair + qkv 实现与 ref 对齐
// ---------------------------------------------------------------------------
TEST (matvec_f16_cuda_fused_pair_qkv_match_ref) {
    if (!set_matvec_f16_impl_by_name("cuda_resident_coal_ws")) {
        std::printf("[skip] current build has no f16 'cuda_resident_coal_ws' matvec (no CUDA toolchain)\n");
        return;
    }

    auto fill = [](std::vector<uint16_t> &w) {
        for (int i = 0; i < static_cast<int>(w.size()); ++i)
            w[i] = float_to_half(static_cast<float>((i * 37 % 29) - 14) * 0.125f);
    };

    // pair：y1 = W1@x，y2 = W2@x（k/v_proj 真实形状）
    {
        const int out_dim = 128, in_dim = 896;
        std::vector<uint16_t> w1((size_t)out_dim * in_dim), w2((size_t)out_dim * in_dim);
        std::vector<float> x(in_dim);
        fill(w1); fill(w2);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> y1(out_dim), y2(out_dim), r1(out_dim), r2(out_dim);
        matvec_f16_ref(w1.data(), x.data(), r1.data(), out_dim, in_dim);
        matvec_f16_ref(w2.data(), x.data(), r2.data(), out_dim, in_dim);
        for (int rep = 0; rep < 3; ++rep) {
            matvec_pair_f16(w1.data(), w2.data(), x.data(), y1.data(), y2.data(), out_dim, in_dim);
            for (int o = 0; o < out_dim; ++o) {
                EXPECT_NEAR(y1[o], r1[o], 5e-3);
                EXPECT_NEAR(y2[o], r2[o], 5e-3);
            }
        }
    }

    // qkv：yq = Wq@x（q_dim），yk/yv = Wk/Wv@x（kv_dim）
    {
        const int q_dim = 896, kv_dim = 128, in_dim = 896; // Qwen2.5-0.5B 真实形状
        std::vector<uint16_t> wq((size_t)q_dim * in_dim), wk((size_t)kv_dim * in_dim),
                wv((size_t)kv_dim * in_dim);
        std::vector<float> x(in_dim);
        fill(wq); fill(wk); fill(wv);
        for (int i = 0; i < in_dim; ++i) x[i] = static_cast<float>((i * 13 % 17) - 8) * 0.125f;
        std::vector<float> yq(q_dim), yk(kv_dim), yv(kv_dim), rq(q_dim), rk(kv_dim), rv(kv_dim);
        matvec_f16_ref(wq.data(), x.data(), rq.data(), q_dim, in_dim);
        matvec_f16_ref(wk.data(), x.data(), rk.data(), kv_dim, in_dim);
        matvec_f16_ref(wv.data(), x.data(), rv.data(), kv_dim, in_dim);
        for (int rep = 0; rep < 3; ++rep) {
            matvec_qkv_f16(wq.data(), wk.data(), wv.data(), x.data(), yq.data(), yk.data(),
                           yv.data(), q_dim, kv_dim, in_dim);
            for (int o = 0; o < q_dim; ++o) EXPECT_NEAR(yq[o], rq[o], 5e-3);
            for (int o = 0; o < kv_dim; ++o) {
                EXPECT_NEAR(yk[o], rk[o], 5e-3);
                EXPECT_NEAR(yv[o], rv[o], 5e-3);
            }
        }
    }

    EXPECT_TRUE(set_matvec_f16_impl_by_name("ref")); // 恢复默认
}

// =========================================================================
//  INT4 量化内核测试
// =========================================================================

// ---------------------------------------------------------------------------
// 测试：INT4 已知值手工计算
//
// 构造一个 [2, 4] 的 INT4 packed 矩阵（group_size=4）。
// 每个 group：[scale_fp16(2B) | zero_fp16(2B) | packed(2B)]
// group_total_bytes = 6, row_bytes = 6, 总共 12 bytes
//
// Row 0: scale=2.0, zero=1.0, vals=[3,5,0,15] -> dequant: [4,8,-2,28]
// Row 1: scale=1.0, zero=0.0, vals=[1,2,3,4] -> dequant: [1,2,3,4]
// x = [1,1,1,1]
// y[0] = 4+8-2+28 = 38, y[1] = 1+2+3+4 = 10
// ---------------------------------------------------------------------------
TEST (matvec_i4_ref_known_values) {
    const int out_dim = 2, in_dim = 4, group_size = 4;
    uint8_t w[12]; // 2 rows * 6 bytes

    // Row 0: scale=2.0(fp16=0x4000), zero=1.0(fp16=0x3C00)
    // vals: [3, 5, 0, 15] -> packed: byte0 = (5<<4)|3 = 0x53, byte1 = (15<<4)|0 = 0xF0
    uint16_t scale_h = 0x4000; // fp16 for 2.0
    uint16_t zero_h = 0x3C00;  // fp16 for 1.0
    std::memcpy(w + 0, &scale_h, 2);
    std::memcpy(w + 2, &zero_h, 2);
    w[4] = 0x53; // low=3, high=5
    w[5] = 0xF0; // low=0, high=15

    // Row 1: scale=1.0(fp16=0x3C00), zero=0.0(fp16=0x0000)
    // vals: [1, 2, 3, 4] -> dequant: [1, 2, 3, 4]
    scale_h = 0x3C00; // fp16 for 1.0
    zero_h = 0x0000;  // fp16 for 0.0
    std::memcpy(w + 6, &scale_h, 2);
    std::memcpy(w + 8, &zero_h, 2);
    w[10] = 0x21; // low=1, high=2
    w[11] = 0x43; // low=3, high=4

    float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[2] = {0};

    matvec_i4(w, x, y, out_dim, in_dim, group_size);

    // row0: (3-1)*2 + (5-1)*2 + (0-1)*2 + (15-1)*2 = 4+8-2+28 = 38
    EXPECT_NEAR(y[0], 38.0f, 1e-3);
    // row1: (1-0)*1 + (2-0)*1 + (3-0)*1 + (4-0)*1 = 10
    EXPECT_NEAR(y[1], 10.0f, 1e-3);
}

// ---------------------------------------------------------------------------
// 测试：INT4 多 group 手动计算
//
// [1, 8] 矩阵，group_size=4 -> 2 groups per row
// Group 0: scale=1.0, zero=0.0, vals=[1,1,1,1] -> sum=4
// Group 1: scale=0.5, zero=2.0, vals=[4,6,8,10] -> dequant=[1,2,3,4] -> sum=10
// Total = 4 + 10 = 14
// ---------------------------------------------------------------------------
TEST (matvec_i4_ref_multi_group) {
    const int out_dim = 1, in_dim = 8, group_size = 4;
    uint8_t w[12]; // 1 row * 12 bytes

    // Group 0: scale=1.0, zero=0.0, vals=[1,1,1,1]
    uint16_t s = 0x3C00, z = 0x0000;
    std::memcpy(w + 0, &s, 2);
    std::memcpy(w + 2, &z, 2);
    w[4] = 0x11; // low=1, high=1
    w[5] = 0x11; // low=1, high=1

    // Group 1: scale=0.5(fp16=0x3800), zero=2.0(fp16=0x4000), vals=[4,6,8,10]
    s = 0x3800; z = 0x4000;
    std::memcpy(w + 6, &s, 2);
    std::memcpy(w + 8, &z, 2);
    w[10] = (6 << 4) | 4;   // low=4, high=6
    w[11] = (10 << 4) | 8;  // low=8, high=10

    float x[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float y[1] = {0};

    matvec_i4(w, x, y, out_dim, in_dim, group_size);

    // group0: 1+1+1+1 = 4; group1: 1+2+3+4 = 10; total = 14
    EXPECT_NEAR(y[0], 14.0f, 1e-3);
}

// ---------------------------------------------------------------------------
// 测试：INT4 dispatch 系统
// ---------------------------------------------------------------------------
TEST (matvec_i4_dispatch_selects_impl) {
    EXPECT_TRUE(set_matvec_i4_impl_by_name("ref"));
    EXPECT_TRUE(!set_matvec_i4_impl_by_name("nonexistent_i4_impl"));
}

// ---------------------------------------------------------------------------
// 测试：INT4 NEON 变体与 ref 对齐
//
// [out=64, in=256], group_size=128 -> 2 groups per row
// ---------------------------------------------------------------------------
TEST (matvec_i4_neon_matches_ref) {
    if (!set_matvec_i4_impl_by_name("neon")) {
        return; // 非 ARM 平台，跳过
    }
    const int out_dim = 64, in_dim = 256, group_size = 128;
    const int group_data = group_size / 2;    // 64 bytes of packed data
    const int group_total = 4 + group_data;   // 68 bytes per group
    const int groups_per_row = in_dim / group_size; // 2 groups per row
    const int row_bytes = groups_per_row * group_total; // 136 bytes per row

    std::vector<uint8_t> w(out_dim * row_bytes);
    std::vector<float> x(in_dim);
    std::vector<float> y_neon(out_dim);
    std::vector<float> y_ref(out_dim);

    // 确定性伪随机填充（LCG 算法）
    uint32_t seed = 42;
    auto rng = [&]() -> uint32_t {
        seed = seed * 1664525u + 1013904223u;
        return seed;
    };

    for (int i = 0; i < in_dim; ++i) {
        x[i] = static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f;
    }

    uint16_t scale_h = 0x2E66; // ≈0.1 in fp16
    uint16_t zero_h = 0x4800;  // 8.0 in fp16
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            uint8_t *gp = w.data() + o * row_bytes + g * group_total;
            std::memcpy(gp, &scale_h, 2);
            std::memcpy(gp + 2, &zero_h, 2);
            for (int b = 0; b < group_data; ++b) {
                gp[4 + b] = static_cast<uint8_t>(rng() & 0xFF);
            }
        }
    }

    // NEON 实现
    set_matvec_i4_impl_by_name("neon");
    matvec_i4(w.data(), x.data(), y_neon.data(), out_dim, in_dim, group_size);

    // Ref 实现
    set_matvec_i4_impl_by_name("ref");
    matvec_i4(w.data(), x.data(), y_ref.data(), out_dim, in_dim, group_size);

    for (int i = 0; i < out_dim; ++i) {
        float tol = std::abs(y_ref[i]) * 1e-4f + 1e-5f;
        EXPECT_NEAR(y_neon[i], y_ref[i], tol);
    }
}

// ---------------------------------------------------------------------------
// 测试：matmul 与独立 matvec 对比
//
// Y[M,N] = W[M,K] × X[K,N] 应该等价于 N 次独立的 matvec。
// ---------------------------------------------------------------------------
TEST (matmul_ref_matches_iterated_matvec) {
    const int M = 8, K = 16, N = 4;
    std::vector<float> w(M * K), x(K * N), y_mm(M * N), y_mv(M * N);

    // 确定性伪随机数据
    uint32_t seed = 123;
    auto rng = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(static_cast<int>(seed % 200) - 100) * 0.01f;
    };
    for (auto &v : w) v = rng();
    for (auto &v : x) v = rng();

    // matmul 实现
    matmul_f32(w.data(), x.data(), y_mm.data(), M, K, N);

    // N 次独立的 matvec（逐列计算）
    for (int c = 0; c < N; ++c) {
        matvec_f32(w.data(), x.data() + c * K, y_mv.data() + c * M, M, K);
    }

    // 逐元素对比
    for (int i = 0; i < M * N; ++i) {
        EXPECT_NEAR(y_mm[i], y_mv[i], 1e-5);
    }
}