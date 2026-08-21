// ============================================================================
// test_gdn_ops.cpp — Qwen3.5 GDN 算子的单元测试
// ============================================================================
// 本文件覆盖 Gated Delta Network (GDN) 相关算子的正确性验证，包括：
//   - l2norm_inplace_ref      — L2 归一化（就地）
//   - causal_conv1d_update_ref — 因果 1D 卷积 + SiLU 激活（单步更新）
//   - gdn_step_ref            — Gated Delta Rule 单步状态更新
//   - rmsnorm_gated_ref       — 带门控的 RMSNorm
//   - partial_rope_ref        — 部分维度的旋转位置编码
//
// 重点覆盖 causal_conv1d 的就地（x/out 同址）状态推进——这是 decode 多步
// 正确性的关键，曾有覆盖 bug（state 存了 silu 后的值而非原始输入）。
//
// 所有测试使用 _ref 后缀的参考实现，确保数值锚点独立于优化变体。
// ============================================================================

#include "test_framework.h"  // 自研测试框架：TEST 宏、EXPECT_NEAR 等

#include <cmath>     // std::exp, std::sqrt, std::fabs
#include <vector>    // std::vector（动态数组）

#include "gdn_ops.h"  // GDN 算子声明（l2norm, conv1d, gdn_step, rmsnorm, rope）

using namespace tinyqwen;  // 引入 tinyqwen 命名空间

namespace {
    // =========================================================================
    // silu() — SiLU 激活函数（Sigmoid Linear Unit）
    // =========================================================================
    // 公式：silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // 用于 conv1d 和 rmsnorm_gated 的期望值计算
    float silu(float x) { return x / (1.0f + std::exp(-x)); }
}

// ---------------------------------------------------------------------------
// 测试：l2norm 单位向量验证
//
// 构造 x = [3, 4, 0]，L2 范数 = sqrt(9+16+0) = 5
// 归一化后应为 [0.6, 0.8, 0]
// eps=0.0f：零向量时除以 0 的行为由其他测试覆盖
// ---------------------------------------------------------------------------
TEST (l2norm_unit) {
    float x[3] = {3.0f, 4.0f, 0.0f};  // 经典 3-4-5 直角三角形比例
    l2norm_inplace_ref(x, 3, 0.0f);    // 就地 L2 归一化，eps=0
    // 验证归一化结果：每个分量 / 范数(5)
    EXPECT_NEAR(x[0], 0.6f, 1e-6);  // 3/5 = 0.6
    EXPECT_NEAR(x[1], 0.8f, 1e-6);  // 4/5 = 0.8
    EXPECT_NEAR(x[2], 0.0f, 1e-6);  // 0/5 = 0
}

// ---------------------------------------------------------------------------
// 测试：l2norm 零向量 + eps 避免除零
//
// 当输入全为零时，范数 = 0，不加 eps 会导致 NaN/Inf。
// eps=1e-6 使分母变为 1e-6，输出应为有限值。
// ---------------------------------------------------------------------------
TEST (l2norm_eps_avoids_div0) {
    float x[2] = {0.0f, 0.0f};      // 零向量
    l2norm_inplace_ref(x, 2, 1e-6f); // eps=1e-6 防止除零
    // 验证结果是有限值（非 NaN、非 Inf）
    EXPECT_TRUE(std::isfinite(x[0]) && std::isfinite(x[1]));
}

// ---------------------------------------------------------------------------
// 测试：conv1d 单步、非就地模式
//
// 验证 kernel_size=4 的因果卷积 + SiLU 激活的正确性。
// 构造 dim=1（单通道），权重 w=[1,2,3,4]，输入序列恒为 1。
//
// 手动推演（state 初始全零）：
//   step0: ctx=[0,0,0,1] → dot=4*1=4         → silu(4)
//   step1: ctx=[0,0,1,1] → dot=3*1+4*1=7     → silu(7)
//   step2: ctx=[0,1,1,1] → dot=2+3+4=9       → silu(9)
//   step3: ctx=[1,1,1,1] → dot=1+2+3+4=10    → silu(10)
//
// 每步 state 自动滑动窗口更新（旧值左移，新输入追加到末尾）。
// ---------------------------------------------------------------------------
TEST (conv1d_step_basic) {
    const int dim = 1, K = 4;                        // 单通道，kernel 大小 4
    std::vector<float> state(dim * (K - 1), 0.0f);   // state 长度 = K-1 = 3，初始全零
    const float w[K] = {1.0f, 2.0f, 3.0f, 4.0f};    // 卷积权重
    float out;                                        // 输出标量
    float x = 1.0f;                                   // 输入恒为 1

    // step0: state=[0,0,0], 新输入=1 → ctx=[0,0,0,1] → dot=4 → silu(4)
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(4.0f), 1e-6);

    // step1: state=[0,0,1], 新输入=1 → ctx=[0,0,1,1] → dot=3+4=7 → silu(7)
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(7.0f), 1e-6);

    // step2: state=[0,1,1], 新输入=1 → ctx=[0,1,1,1] → dot=2+3+4=9 → silu(9)
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(9.0f), 1e-6);

    // step3: state=[1,1,1], 新输入=1 → ctx=[1,1,1,1] → dot=1+2+3+4=10 → silu(10)
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(10.0f), 1e-6);
}

// ---------------------------------------------------------------------------
// 测试：conv1d 就地模式（x 与 out 同址）state 不被污染
//
// 【关键回归测试】此前有 bug：先写 out[c]=silu(...) 再把 x[c]（已被 silu
// 覆盖）推进 state，导致后续步骤使用了错误的历史值。
//
// 构造 dim=2, K=3（state_len=2），权重 w={0,5,5} 使卷积结果与原始输入
// 差异大，放大"存错值"的影响。
//
// step0: buf=[1,2], state=[[0,0],[0,0]]
//   ch0: dot = 0*0 + 5*0 + 5*1 = 5 → silu(5) ≈ 4.97
//   ch1: dot = 0*0 + 5*0 + 5*2 = 10 → silu(10) ≈ 9.9995
//   state 应存原始输入 [[0,1],[0,2]]，而非 silu 值
//
// step1: buf=[1,2], state=[[0,1],[0,2]]
//   ch0: dot = 0*0 + 5*1 + 5*1 = 10 → silu(10)
//   ch1: dot = 0*0 + 5*2 + 5*2 = 20 → silu(20)
// ---------------------------------------------------------------------------
TEST (conv1d_inplace_state_not_corrupted) {
    const int dim = 2, K = 3;                           // 双通道，kernel=3
    std::vector<float> state(dim * (K - 1), 0.0f);      // state 初始全零
    // 权重设计：w[0]=0 忽略最旧值，w[1]=w[2]=5 放大最新两个值的贡献
    std::vector<float> w = {0.0f, 5.0f, 5.0f,           // ch0 权重
                            0.0f, 5.0f, 5.0f};          // ch1 权重
    std::vector<float> buf = {1.0f, 2.0f};              // 就地 buffer（同时作 x 和 out）

    // step0: 就地调用，buf 被 silu 覆盖
    causal_conv1d_update_ref(buf.data(), state.data(), w.data(), buf.data(), dim, K);
    // 验证输出是 silu 值
    EXPECT_NEAR(buf[0], silu(5.0f), 1e-6);    // ch0: silu(5*1)
    EXPECT_NEAR(buf[1], silu(10.0f), 1e-6);   // ch1: silu(5*2)
    // 【关键断言】state 必须存原始输入，不是 silu 值
    EXPECT_NEAR(state[0], 0.0f, 1e-6);        // ch0 旧 state[0] = 0
    EXPECT_NEAR(state[1], 1.0f, 1e-6);        // ch0 旧 state[1] = 原始输入 1（非 silu(5)）
    EXPECT_NEAR(state[2], 0.0f, 1e-6);        // ch1 旧 state[0] = 0
    EXPECT_NEAR(state[3], 2.0f, 1e-6);        // ch1 旧 state[1] = 原始输入 2（非 silu(10)）

    // step1: 再喂同样的 [1,2]，此时 state=[[0,1],[0,2]]
    buf = {1.0f, 2.0f};
    causal_conv1d_update_ref(buf.data(), state.data(), w.data(), buf.data(), dim, K);
    // ch0: dot = 5*(1+1) = 10 → silu(10)
    EXPECT_NEAR(buf[0], silu(5.0f * (1.0f + 1.0f)), 1e-5);
    // ch1: dot = 5*(2+2) = 20 → silu(20)
    EXPECT_NEAR(buf[1], silu(5.0f * (2.0f + 2.0f)), 1e-5);
}

// ---------------------------------------------------------------------------
// 测试：gdn_step 从零状态出发的单步手工验证
//
// Gated Delta Rule 的核心公式：
//   decay = exp(g)，kv_mem = S^T q，delta = beta*(v - kv_mem)
//   S = decay*S + outer(k, delta)，o = S^T q
//
// 构造 q=[1,0], k=[0,1], v=[3,4], g=0(decay=1), beta=1, S 初始全零。
//
// 推演：
//   kv_mem = S^T q = [0,0]（S 全零）
//   delta = 1*(v - kv_mem) = [3,4]
//   S = outer(k, delta) = [[0*3, 0*4], [1*3, 1*4]] = [[0,0],[3,4]]
//   o = S^T q = [0*1+3*0, 0*1+4*0] = [0, 0]
//   （因为 q 和 k 正交，所以 o = 0）
// ---------------------------------------------------------------------------
TEST (gdn_step_from_zero_state) {
    const int qk = 2, vd = 2;                      // qk_dim=2, v_dim=2
    std::vector<float> S(qk * vd, 0.0f);           // 状态矩阵 S（2×2，初始全零）
    const float q[2] = {1.0f, 0.0f};               // query 向量
    const float k[2] = {0.0f, 1.0f};               // key 向量（与 q 正交）
    const float v[2] = {3.0f, 4.0f};               // value 向量
    float o[2];                                      // 输出向量

    // g=0 → decay=exp(0)=1；beta=1（完全写入）
    gdn_step_ref(S.data(), q, k, v, 0.0f, 1.0f, o, qk, vd);

    // o = S^T q = [0, 0]（q 和 k 正交的结果）
    EXPECT_NEAR(o[0], 0.0f, 1e-6);
    EXPECT_NEAR(o[1], 0.0f, 1e-6);
    // 验证状态已被正确写入：S[j][d] = k[j]*delta[d]
    // S[1][0] = k[1]*delta[0] = 1*3 = 3
    EXPECT_NEAR(S[1 * vd + 0], 3.0f, 1e-6);
    // S[1][1] = k[1]*delta[1] = 1*4 = 4
    EXPECT_NEAR(S[1 * vd + 1], 4.0f, 1e-6);
}

// ---------------------------------------------------------------------------
// 测试：gdn_step 写入后立即读回（q==k 时的直觉校验）
//
// 当 q == k 且 decay=1, beta=1 时：
//   kv_mem = S^T k = 0（S 初始为零）
//   delta = v - 0 = v
//   S = outer(k, v)
//   o = S^T k = outer(k,v)^T k = v * (k·k) = v * ||k||^2
//   当 k 为单位向量 [1,0] 时，||k||^2=1，故 o = v
// ---------------------------------------------------------------------------
TEST (gdn_step_readback) {
    const int qk = 2, vd = 2;                      // 维度参数
    std::vector<float> S(qk * vd, 0.0f);           // 状态矩阵初始全零
    const float k[2] = {1.0f, 0.0f};               // key = 单位基向量 e1
    const float v[2] = {2.0f, -1.0f};              // value 向量
    float o[2];                                      // 输出

    // beta=1, decay=1：用同一个 k 既写又读
    gdn_step_ref(S.data(), k, k, v, 0.0f, 1.0f, o, qk, vd);

    // o 应等于 v（写入后立即用相同 key 读回）
    EXPECT_NEAR(o[0], v[0], 1e-5);  // 2.0
    EXPECT_NEAR(o[1], v[1], 1e-5);  // -1.0
}

// ---------------------------------------------------------------------------
// 测试：rmsnorm_gated 与数学公式逐元素对照
//
// RMSNorm gated 公式：
//   y[i] = (x[i] / RMS(x)) * weight[i] * silu(gate[i])
//   RMS(x) = sqrt(mean(x^2) + eps)
//
// 构造 x=[3,4], gate=[0,1], w=[1,2], eps=0
//   RMS = sqrt((9+16)/2) = sqrt(12.5)
//   y[0] = (3/sqrt(12.5)) * 1 * silu(0) = (3/sqrt(12.5)) * 1 * 0 = 0
//   y[1] = (4/sqrt(12.5)) * 2 * silu(1)
// ---------------------------------------------------------------------------
TEST (rmsnorm_gated_matches_formula) {
    const float x[2] = {3.0f, 4.0f};          // 输入向量
    const float gate[2] = {0.0f, 1.0f};       // 门控向量
    const float w[2] = {1.0f, 2.0f};          // 缩放权重
    float y[2];                                  // 输出

    rmsnorm_gated_ref(x, gate, w, y, 2, 0.0f); // eps=0

    const float rms = std::sqrt(12.5f);         // RMS = sqrt((9+16)/2)
    // y[0] = (3/rms) * 1.0 * silu(0.0)
    EXPECT_NEAR(y[0], (3.0f / rms) * 1.0f * silu(0.0f), 1e-5);
    // y[1] = (4/rms) * 2.0 * silu(1.0)
    EXPECT_NEAR(y[1], (4.0f / rms) * 2.0f * silu(1.0f), 1e-5);
}

// ---------------------------------------------------------------------------
// 测试：partial_rope 在 pos=0 时是恒等变换
//
// RoPE 在 pos=0 时，所有旋转角 = 0，cos=1, sin=0，等价于恒等映射。
// 同时验证 rotary_dim 之外的维度（尾部）保持不变（passthrough）。
// ---------------------------------------------------------------------------
TEST (partial_rope_pos0_identity_and_passthrough) {
    const int n_heads = 1, n_kv = 1, head_dim = 8, rotary = 4;  // 只旋转前 4 维
    std::vector<float> q(head_dim), k(head_dim);
    // 填充非零值：q[i] = 0.1*(i+1), k[i] = -0.05*(i+1)
    for (int i = 0; i < head_dim; ++i) { q[i] = 0.1f * (i + 1); k[i] = -0.05f * (i + 1); }
    // 保存原始值作为参照
    const std::vector<float> q0 = q, k0 = k;
    // pos=0 时应为恒等
    partial_rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, rotary, 0, 10000.0f);
    // 验证所有维度都不变
    for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(q[i], q0[i], 1e-6);
        EXPECT_NEAR(k[i], k0[i], 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 测试：partial_rope 对 rotary_dim 之外的维度 passthrough
//
// 验证：
//   1. 尾部 [rotary, head_dim) 在任何 pos 下都保持不变
//   2. 旋转部分 [0, rotary) 在 pos!=0 时确实发生变化
// ---------------------------------------------------------------------------
TEST (partial_rope_passes_tail) {
    const int n_heads = 1, n_kv = 1, head_dim = 8, rotary = 4;
    std::vector<float> q(head_dim, 0.0f), k(head_dim, 0.0f);
    // 全部填 1.0，便于检测变化
    for (int i = 0; i < head_dim; ++i) { q[i] = 1.0f; k[i] = 1.0f; }
    // pos=5 时执行旋转
    partial_rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, rotary, 5, 10000.0f);
    // 尾部 [4, 8) 不应被旋转修改
    for (int i = rotary; i < head_dim; ++i) {
        EXPECT_NEAR(q[i], 1.0f, 1e-6);
        EXPECT_NEAR(k[i], 1.0f, 1e-6);
    }
    // 旋转部分 [0, 4) 在 pos=5 时不全为 1（确认旋转确实生效）
    bool changed = false;
    for (int i = 0; i < rotary; ++i) if (std::fabs(q[i] - 1.0f) > 1e-3) changed = true;
    EXPECT_TRUE(changed);  // 至少有一个维度发生了变化
}
