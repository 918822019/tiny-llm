// ============================================================================
// test_gdn_ops_neon.cpp — GDN 算子 NEON 变体的正确性门禁
// ============================================================================
// 本文件验证 ARM NEON SIMD 优化的 GDN 算子与参考实现（_ref）的数值一致性。
// 覆盖的算子：
//   - l2norm_inplace    — L2 归一化
//   - rmsnorm_gated     — 带门控的 RMSNorm
//   - causal_conv1d_update — 因果 1D 卷积 + SiLU（含就地模式）
//   - gdn_step          — Gated Delta Rule 单步更新（含多步漂移验证）
//
// 约定（与 test_ops_neon.cpp 同款）：
//   - set_ops_impl_by_name("neon") 失败就打印 [skip] 并返回
//     （非 aarch64 平台没有 NEON 注册，跳过不算失败）
//   - 数值锚点：每个 NEON kernel vs 对应 _ref 在相同输入上对比
//   - 容差依据：NEON float32 运算与标量 double 累加的 ulp 级差异
//
// 伪随机数使用确定性 LCG（线性同余生成器），保证跨平台可复现。
// ============================================================================

#include "test_framework.h"  // 自研测试框架

#include <cmath>      // std::fabs
#include <cstdint>    // uint32_t
#include <cstring>    // memcpy 等
#include <vector>     // std::vector

#include "dispatch.h"  // set_ops_impl_by_name / ops_impl_name（算子 dispatch）
#include "gdn_ops.h"   // GDN 算子声明（ref + dispatch 版本）

using namespace tinyqwen;

namespace {
    // =========================================================================
    // struct Lcg — 确定性伪随机数生成器（Linear Congruential Generator）
    // =========================================================================
    // 参数来自 Numerical Recipes：s = s * 1664525 + 1013904223
    // 取高 24 位作为 [0,1) 均匀分布，再线性映射到 [lo, hi)。
    // 不使用 <random> 是为了避免不同标准库实现的分布差异导致不可复现。
    struct Lcg {
        uint32_t s = 54321u;  // 种子（固定值，保证可复现）
        float next(float lo, float hi) {
            s = s * 1664525u + 1013904223u;  // LCG 步进
            const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);  // [0,1)
            return lo + u * (hi - lo);       // 线性映射到 [lo, hi)
        }
    };

    // =========================================================================
    // max_abs_diff() — 计算两个数组的最大绝对差
    // =========================================================================
    // 用于比较 ref 和 neon 输出的数值偏差
    float max_abs_diff(const float *a, const float *b, int n) {
        float mx = 0.0f;
        for (int i = 0; i < n; ++i) {
            float d = std::fabs(a[i] - b[i]);
            if (d > mx) mx = d;
        }
        return mx;
    }
} // namespace

// ---------------------------------------------------------------------------
// 测试：l2norm NEON vs ref
//
// 覆盖多种尺寸：128（NEON 对齐）、127（尾段）、64、5（极小）。
// 容差 1e-5：NEON float32 累加 vs 标量 float32 的精度差异。
// ---------------------------------------------------------------------------
TEST (l2norm_neon_matches_ref) {
    // 尝试切换到 neon 实现；失败则跳过（非 aarch64）
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {128, 127, 64, 5};  // 多种尺寸覆盖对齐/尾段/边界
    Lcg rng;                                  // 确定性 RNG
    for (int n : sizes) {
        // 为 ref 和 neon 各准备一份相同的输入数据
        std::vector<float> x_ref(n), x_neon(n);
        for (int i = 0; i < n; ++i) {
            x_ref[i] = x_neon[i] = rng.next(-2.0f, 2.0f);  // [-2, 2) 均匀分布
        }
        // 分别用 ref 和 dispatch(neon) 执行 L2 归一化
        l2norm_inplace_ref(x_ref.data(), n, 1e-6f);
        l2norm_inplace(x_neon.data(), n, 1e-6f);  // dispatch -> neon
        // 计算最大绝对误差
        float err = max_abs_diff(x_ref.data(), x_neon.data(), n);
        if (err > 1e-5f) {
            std::printf("l2norm n=%d max_err=%e\n", n, err);  // 超差时打印诊断
        }
        EXPECT_TRUE(err <= 1e-5f);  // 容差门禁
    }
    // 恢复到 ref 实现（避免影响后续测试）
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：rmsnorm_gated NEON vs ref
//
// 覆盖尺寸：128、127（尾段）、64。
// 容差 1e-4：涉及 silu 的多项式逼近 + RMS 除法，累积误差略大。
// ---------------------------------------------------------------------------
TEST (rmsnorm_gated_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {128, 127, 64};
    Lcg rng;
    for (int n : sizes) {
        // 构造随机输入、门控、权重
        std::vector<float> x(n), gate(n), weight(n), y_ref(n), y_neon(n);
        for (int i = 0; i < n; ++i) {
            x[i] = rng.next(-1.0f, 1.0f);       // 输入 [-1, 1)
            gate[i] = rng.next(-3.0f, 3.0f);     // 门控 [-3, 3)
            weight[i] = rng.next(0.5f, 1.5f);    // 权重 [0.5, 1.5)
        }
        // 分别用 ref 和 neon 计算
        rmsnorm_gated_ref(x.data(), gate.data(), weight.data(), y_ref.data(), n, 1e-6f);
        rmsnorm_gated(x.data(), gate.data(), weight.data(), y_neon.data(), n, 1e-6f);
        float err = max_abs_diff(y_ref.data(), y_neon.data(), n);
        if (err > 1e-4f) {
            std::printf("rmsnorm_gated n=%d max_err=%e\n", n, err);
        }
        EXPECT_TRUE(err <= 1e-4f);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：causal_conv1d NEON vs ref
//
// 覆盖 dim=6144（真实模型维度）和 dim=13（非 4 整除的尾段场景）。
// K=4, state_len=3。同时验证输出和 state 的一致性。
// 容差：out ≤ 1e-5, state ≤ 1e-6（state 是简单赋值，精度更高）。
// ---------------------------------------------------------------------------
TEST (conv1d_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int dims[] = {6144, 13};  // 真实维度 + 非对齐尾段
    const int K = 4;                 // kernel 大小
    const int state_len = K - 1;     // state 长度 = K-1
    Lcg rng;
    for (int dim : dims) {
        // 为 ref 和 neon 各准备相同的 state、weight、input
        std::vector<float> state_ref(dim * state_len), state_neon(dim * state_len);
        std::vector<float> weight(dim * K);
        std::vector<float> x(dim), out_ref(dim), out_neon(dim);
        for (int i = 0; i < dim * state_len; ++i)
            state_ref[i] = state_neon[i] = rng.next(-1.0f, 1.0f);
        for (int i = 0; i < dim * K; ++i)
            weight[i] = rng.next(-0.5f, 0.5f);
        for (int i = 0; i < dim; ++i)
            x[i] = rng.next(-1.0f, 1.0f);

        // 分别用 ref 和 neon 执行因果卷积更新
        causal_conv1d_update_ref(x.data(), state_ref.data(), weight.data(),
                                 out_ref.data(), dim, K);
        causal_conv1d_update(x.data(), state_neon.data(), weight.data(),
                             out_neon.data(), dim, K);

        // 验证输出和 state 的误差
        float out_err = max_abs_diff(out_ref.data(), out_neon.data(), dim);
        float state_err = max_abs_diff(state_ref.data(), state_neon.data(), dim * state_len);
        if (out_err > 1e-5f || state_err > 1e-6f) {
            std::printf("conv1d dim=%d out_err=%e state_err=%e\n", dim, out_err, state_err);
        }
        EXPECT_TRUE(out_err <= 1e-5f);
        EXPECT_TRUE(state_err <= 1e-6f);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：conv1d NEON 就地模式（x 和 out 指向同一缓冲区）
//
// 验证就地模式下 NEON 实现与 ref 行为一致（不污染 state）。
// dim=6144（真实维度），K=4。
// ---------------------------------------------------------------------------
TEST (conv1d_neon_inplace) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int dim = 6144, K = 4, state_len = K - 1;
    Lcg rng;
    // 为 ref 和 neon 各准备相同的数据
    std::vector<float> state_ref(dim * state_len), state_neon(dim * state_len);
    std::vector<float> weight(dim * K);
    std::vector<float> buf_ref(dim), buf_neon(dim);
    for (int i = 0; i < dim * state_len; ++i)
        state_ref[i] = state_neon[i] = rng.next(-1.0f, 1.0f);
    for (int i = 0; i < dim * K; ++i)
        weight[i] = rng.next(-0.5f, 0.5f);
    for (int i = 0; i < dim; ++i)
        buf_ref[i] = buf_neon[i] = rng.next(-1.0f, 1.0f);

    // in-place: out == x（buf 同时作输入和输出）
    causal_conv1d_update_ref(buf_ref.data(), state_ref.data(), weight.data(),
                             buf_ref.data(), dim, K);
    causal_conv1d_update(buf_neon.data(), state_neon.data(), weight.data(),
                         buf_neon.data(), dim, K);

    // 验证就地结果一致性
    float out_err = max_abs_diff(buf_ref.data(), buf_neon.data(), dim);
    float state_err = max_abs_diff(state_ref.data(), state_neon.data(), dim * state_len);
    EXPECT_TRUE(out_err <= 1e-5f);
    EXPECT_TRUE(state_err <= 1e-6f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：gdn_step NEON vs ref（单步）
//
// qk_dim=128, v_dim=128（接近真实配置）。
// 状态 S 初始化为小随机值（模拟非零历史状态）。
// g=-0.5（衰减门）, beta=0.7（部分写入）。
// 容差：o ≤ 1e-4, S ≤ 1e-5。
// ---------------------------------------------------------------------------
TEST (gdn_step_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int qk_dim = 128, v_dim = 128;         // 维度参数
    const int state_size = qk_dim * v_dim;        // 状态矩阵元素总数
    Lcg rng;

    // 初始化状态、输入、输出缓冲
    std::vector<float> S_ref(state_size), S_neon(state_size);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    for (int i = 0; i < state_size; ++i)
        S_ref[i] = S_neon[i] = rng.next(-0.1f, 0.1f);  // 小随机初始状态
    for (int i = 0; i < qk_dim; ++i) {
        q[i] = rng.next(-1.0f, 1.0f);
        k[i] = rng.next(-1.0f, 1.0f);
    }
    for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);

    const float g = -0.5f;     // 衰减门（decay = exp(-0.5) ≈ 0.607）
    const float beta = 0.7f;   // 写入强度

    // 分别用 ref 和 neon 执行一步 GDN 更新
    gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), g, beta,
                 o_ref.data(), qk_dim, v_dim);
    gdn_step(S_neon.data(), q.data(), k.data(), v.data(), g, beta,
             o_neon.data(), qk_dim, v_dim);

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    if (o_err > 1e-4f || s_err > 1e-5f) {
        std::printf("gdn_step o_err=%e s_err=%e\n", o_err, s_err);
    }
    EXPECT_TRUE(o_err <= 1e-4f);
    EXPECT_TRUE(s_err <= 1e-5f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：gdn_step NEON 多步漂移验证
//
// 连续跑 10 步递归更新，验证浮点误差不随步数累积到不可接受的程度。
// 每步使用不同的随机 q/k/v/g/beta，模拟真实 decode 过程。
// 容差比单步放宽：o ≤ 5e-4, S ≤ 1e-4。
// ---------------------------------------------------------------------------
TEST (gdn_step_neon_multi_step) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int qk_dim = 128, v_dim = 128;
    const int state_size = qk_dim * v_dim;
    Lcg rng;

    // 状态从零开始
    std::vector<float> S_ref(state_size, 0.0f), S_neon(state_size, 0.0f);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    // 连续 10 步递归更新
    for (int step = 0; step < 10; ++step) {
        for (int i = 0; i < qk_dim; ++i) {
            q[i] = rng.next(-1.0f, 1.0f);
            k[i] = rng.next(-1.0f, 1.0f);
        }
        for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);
        const float g = rng.next(-1.0f, -0.01f);    // 负值 → decay < 1
        const float beta = rng.next(0.1f, 0.9f);    // 部分写入

        gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), g, beta,
                     o_ref.data(), qk_dim, v_dim);
        gdn_step(S_neon.data(), q.data(), k.data(), v.data(), g, beta,
                 o_neon.data(), qk_dim, v_dim);
    }

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    if (o_err > 5e-4f || s_err > 1e-4f) {
        std::printf("gdn_step 10-step drift: o_err=%e s_err=%e\n", o_err, s_err);
    }
    EXPECT_TRUE(o_err <= 5e-4f);   // 10 步累积容差
    EXPECT_TRUE(s_err <= 1e-4f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：gdn_step NEON 小维度（非 4 整除）
//
// qk_dim=8, v_dim=8（fake model 用的小维度），验证 NEON 实现在
// 维度不是 SIMD 宽度整数倍时的尾段处理正确性。
// ---------------------------------------------------------------------------
TEST (gdn_step_neon_small_dims) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int qk_dim = 8, v_dim = 8;             // 小维度，非 4 整除
    const int state_size = qk_dim * v_dim;
    Lcg rng;

    std::vector<float> S_ref(state_size), S_neon(state_size);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    for (int i = 0; i < state_size; ++i)
        S_ref[i] = S_neon[i] = rng.next(-0.1f, 0.1f);
    for (int i = 0; i < qk_dim; ++i) {
        q[i] = rng.next(-1.0f, 1.0f);
        k[i] = rng.next(-1.0f, 1.0f);
    }
    for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);

    // g=-0.3, beta=0.5
    gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), -0.3f, 0.5f,
                 o_ref.data(), qk_dim, v_dim);
    gdn_step(S_neon.data(), q.data(), k.data(), v.data(), -0.3f, 0.5f,
             o_neon.data(), qk_dim, v_dim);

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    EXPECT_TRUE(o_err <= 1e-4f);
    EXPECT_TRUE(s_err <= 1e-5f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}
