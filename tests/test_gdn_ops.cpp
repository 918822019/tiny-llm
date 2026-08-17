// Qwen3.5 GDN 算子的单元测试。重点覆盖 causal_conv1d 的就地（x/out 同址）
// 状态推进——这是 decode 多步正确性的关键，曾有覆盖 bug。

#include "test_framework.h"

#include <cmath>
#include <vector>

#include "gdn_ops.h"

using namespace tinyqwen;

namespace {
    float silu(float x) { return x / (1.0f + std::exp(-x)); }
}

TEST (l2norm_unit) {
    float x[3] = {3.0f, 4.0f, 0.0f};
    l2norm_inplace_ref(x, 3, 0.0f);
    // norm = 5 -> (0.6, 0.8, 0)
    EXPECT_NEAR(x[0], 0.6f, 1e-6);
    EXPECT_NEAR(x[1], 0.8f, 1e-6);
    EXPECT_NEAR(x[2], 0.0f, 1e-6);
}

TEST (l2norm_eps_avoids_div0) {
    float x[2] = {0.0f, 0.0f};
    l2norm_inplace_ref(x, 2, 1e-6f);
    EXPECT_TRUE(std::isfinite(x[0]) && std::isfinite(x[1]));
}

// conv 单步、非就地：验证 kernel=4 的因果卷积 + silu。
TEST (conv1d_step_basic) {
    const int dim = 1, K = 4;
    std::vector<float> state(dim * (K - 1), 0.0f);
    const float w[K] = {1.0f, 2.0f, 3.0f, 4.0f};
    float out;
    // 输入序列 1, 1, 1, 1 ...
    float x = 1.0f;
    // step0: ctx=[0,0,0,1] -> 4*1=4 -> silu(4)
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(4.0f), 1e-6);
    // step1: ctx=[0,0,1,1] -> 3+4=7
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(7.0f), 1e-6);
    // step2: ctx=[0,1,1,1] -> 2+3+4=9
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(9.0f), 1e-6);
    // step3: ctx=[1,1,1,1] -> 1+2+3+4=10
    causal_conv1d_update_ref(&x, state.data(), w, &out, dim, K);
    EXPECT_NEAR(out, silu(10.0f), 1e-6);
}

// 关键回归：x 与 out 同址（就地）时，state 必须存"原始输入"而不是被 silu
// 覆盖后的值。此前 bug：先写 out[c]=silu 再把 x[c]（已被覆盖）推进 state。
TEST (conv1d_inplace_state_not_corrupted) {
    const int dim = 2, K = 3; // state_len = 2
    std::vector<float> state(dim * (K - 1), 0.0f);
    // 权重让卷积结果与输入差异大，放大"存错值"的影响。
    // w = {0, 5, 5}：w[1] 乘最新 state、w[2] 乘当前输入。
    std::vector<float> w = {0.0f, 5.0f, 5.0f,   // ch0
                            0.0f, 5.0f, 5.0f};  // ch1
    std::vector<float> buf = {1.0f, 2.0f}; // 就地 buffer

    // step0: out = silu(5*buf) -> buf 被覆盖成 silu 值
    causal_conv1d_update_ref(buf.data(), state.data(), w.data(), buf.data(), dim, K);
    EXPECT_NEAR(buf[0], silu(5.0f), 1e-6);
    EXPECT_NEAR(buf[1], silu(10.0f), 1e-6);
    // state 必须是原始输入 (0,1)/(0,2)，而不是 silu 值。
    EXPECT_NEAR(state[0], 0.0f, 1e-6);
    EXPECT_NEAR(state[1], 1.0f, 1e-6); // ch1 原始输入 = 1（不是 silu(5)）
    EXPECT_NEAR(state[2], 0.0f, 1e-6);
    EXPECT_NEAR(state[3], 2.0f, 1e-6); // ch2 原始输入 = 2（不是 silu(10)）

    // step1 再喂原始 1,2：ctx=[1,1]/[2,2] -> 5*(1+1)=10, 5*(2+2)=20
    buf = {1.0f, 2.0f};
    causal_conv1d_update_ref(buf.data(), state.data(), w.data(), buf.data(), dim, K);
    EXPECT_NEAR(buf[0], silu(5.0f * (1.0f + 1.0f)), 1e-5);
    EXPECT_NEAR(buf[1], silu(5.0f * (2.0f + 2.0f)), 1e-5);
}

// gated delta rule 单步手工验证（state 初始为 0）。
TEST (gdn_step_from_zero_state) {
    const int qk = 2, vd = 2;
    std::vector<float> S(qk * vd, 0.0f);
    const float q[2] = {1.0f, 0.0f};
    const float k[2] = {0.0f, 1.0f};
    const float v[2] = {3.0f, 4.0f};
    float o[2];
    // g=0 -> decay=1；beta=1。S 从 0：kv_mem=0，delta=v，S=outer(k,v)。
    gdn_step_ref(S.data(), q, k, v, 0.0f, 1.0f, o, qk, vd);
    // S = outer(k, v) = [[0,0],[3,4]]；o = S^T q = [S[0][0]*1+S[1][0]*0, ...]
    // o[d] = sum_j S[j][d]*q[j] = S[0][d]*1 = [0, 0]? 重新算：
    // S[j][d]=k[j]*v[d]. o[d]=sum_j q[j]*S[j][d]=q[0]*k[0]*v[d]+q[1]*k[1]*v[d]
    //      = (1*0 + 0*1)*v[d] = 0. 因为 q 和 k 正交。
    EXPECT_NEAR(o[0], 0.0f, 1e-6);
    EXPECT_NEAR(o[1], 0.0f, 1e-6);
    // 状态已被写入：S[1][0]=k[1]*delta[0]=1*3=3, S[1][1]=4。
    EXPECT_NEAR(S[1 * vd + 0], 3.0f, 1e-6);
    EXPECT_NEAR(S[1 * vd + 1], 4.0f, 1e-6);
}

// q==k 时，一步之后 o 应接近 beta 加权的 v（直觉校验 + 衰减门）。
TEST (gdn_step_readback) {
    const int qk = 2, vd = 2;
    std::vector<float> S(qk * vd, 0.0f);
    const float k[2] = {1.0f, 0.0f};
    const float v[2] = {2.0f, -1.0f};
    float o[2];
    // beta=1, decay=1：写入后立刻用同一个 k 读回 -> o = v。
    gdn_step_ref(S.data(), k, k, v, 0.0f, 1.0f, o, qk, vd);
    EXPECT_NEAR(o[0], v[0], 1e-5);
    EXPECT_NEAR(o[1], v[1], 1e-5);
}

TEST (rmsnorm_gated_matches_formula) {
    const float x[2] = {3.0f, 4.0f};
    const float gate[2] = {0.0f, 1.0f};
    const float w[2] = {1.0f, 2.0f};
    float y[2];
    rmsnorm_gated_ref(x, gate, w, y, 2, 0.0f);
    const float rms = std::sqrt(12.5f);
    EXPECT_NEAR(y[0], (3.0f / rms) * 1.0f * silu(0.0f), 1e-5);
    EXPECT_NEAR(y[1], (4.0f / rms) * 2.0f * silu(1.0f), 1e-5);
}

// partial rope：pos=0 时是恒等；rotary_dim 之外的维度保持不变。
TEST (partial_rope_pos0_identity_and_passthrough) {
    const int n_heads = 1, n_kv = 1, head_dim = 8, rotary = 4;
    std::vector<float> q(head_dim), k(head_dim);
    for (int i = 0; i < head_dim; ++i) { q[i] = 0.1f * (i + 1); k[i] = -0.05f * (i + 1); }
    const std::vector<float> q0 = q, k0 = k;
    partial_rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, rotary, 0, 10000.0f);
    for (int i = 0; i < head_dim; ++i) {
        EXPECT_NEAR(q[i], q0[i], 1e-6);
        EXPECT_NEAR(k[i], k0[i], 1e-6);
    }
}

TEST (partial_rope_passes_tail) {
    const int n_heads = 1, n_kv = 1, head_dim = 8, rotary = 4;
    std::vector<float> q(head_dim, 0.0f), k(head_dim, 0.0f);
    for (int i = 0; i < head_dim; ++i) { q[i] = 1.0f; k[i] = 1.0f; }
    partial_rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, rotary, 5, 10000.0f);
    // 尾部 [rotary, head_dim) 不被旋转。
    for (int i = rotary; i < head_dim; ++i) {
        EXPECT_NEAR(q[i], 1.0f, 1e-6);
        EXPECT_NEAR(k[i], 1.0f, 1e-6);
    }
    // 旋转部分在 pos!=0 时发生变化（不全为 1）。
    bool changed = false;
    for (int i = 0; i < rotary; ++i) if (std::fabs(q[i] - 1.0f) > 1e-3) changed = true;
    EXPECT_TRUE(changed);
}
