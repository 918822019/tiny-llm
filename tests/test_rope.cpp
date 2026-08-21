// ============================================================================
// test_rope.cpp — RoPE（Rotary Position Embedding）旋转位置编码的单元测试
// ============================================================================
// 本文件测试 rope_ref 实现的正确性，包括：
//   1. 单头位置 1 的手动计算验证
//   2. 旋转是正交变换 —— 模长必须保持不变
//   3. GQA 场景下每个 head 独立旋转
//   4. pos=0 时 RoPE 应为恒等变换
//
// RoPE 是 LLM 中广泛使用的位置编码方法，通过将 Q/K 向量按维度配对
// 旋转，将相对位置信息编码到 attention score 中。
//
// 旋转公式（对第 d 对维度）：
//   cos_theta = cos(pos * inv_freq[d])
//   sin_theta = sin(pos * inv_freq[d])
//   q[d]' = q[d] * cos_theta - q[d+1] * sin_theta
//   q[d+1]' = q[d] * sin_theta + q[d+1] * cos_theta
// 其中 inv_freq[d] = 1 / theta^(2*d/head_dim)
// ============================================================================

#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

// ---------------------------------------------------------------------------
// 测试：单头位置 1 的手动计算验证
//
// head_dim=4、pos=1、theta=10000
// dim=0: inv_freq = 1/10000^(0/4) = 1.0，angle = 1*1.0 = 1.0
// dim=2: inv_freq = 1/10000^(2/4) = 0.01，angle = 1*0.01 = 0.01
//
// 对 q=[1,0,0,0]：
//   q[0] = 1*cos(1) - 0*sin(1) = cos(1)
//   q[2] = 0*cos(0.01) - 0*sin(0.01) = 0（但 RoPE 对 dim 2,3 也旋转）
// 实际上每个 2D 配对都旋转，dim 0-1 配对，dim 2-3 配对
// ---------------------------------------------------------------------------
TEST (rope_single_head_pos1) {
    // head_dim=4、pos=1、theta=10000 时 inv_freq = [1.0, 0.01]，可手算验证。
    float q[4] = {1.0f, 0.0f, 0.0f, 0.0f}; // Q 向量 [dim0, dim1, dim2, dim3]
    float k[4] = {0.0f, 1.0f, 0.0f, 0.0f}; // K 向量

    // 调用 RoPE：n_heads=1, n_kv=1, head_dim=4, pos=1, theta=10000
    rope_ref(q, k, 1, 1, 4, 1, 10000.0f);

    // dim 0-1 配对：angle = 1.0
    // q[0] = 1*cos(1) - 0*sin(1) = cos(1)
    EXPECT_NEAR(q[0], std::cos(1.0), 1e-6);
    // q[1] = 1*sin(1) + 0*cos(1) = sin(1)（但 q[1] 初始为 0，旋转后仍为 0？不对）
    // 实际上：q[1] = q[0]*sin(1) + q[1]*cos(1) = 1*sin(1) + 0*cos(1) = sin(1)
    EXPECT_NEAR(q[1], 0.0, 1e-6); // 等等，这里为什么是 0.0？
    // 重新看：q[0] 初始为 1.0，q[1] 初始为 0.0
    // q[0]' = 1.0*cos(1) - 0.0*sin(1) = cos(1)
    // q[1]' = 1.0*sin(1) + 0.0*cos(1) = sin(1)
    // 但代码期望 q[1] = 0.0，说明我理解有误...
    // 实际上 RoPE 是对 dim 0-1 配对，dim 2-3 配对，但 q[0] 和 q[1] 配对
    // 不对，看代码期望：q[0]=cos(1), q[1]=0, q[2]=sin(1), q[3]=0
    // 这说明 RoPE 的配对是 (0,2) 和 (1,3)，不是 (0,1) 和 (2,3)
    // 实际上 RoPE 把前半部分和后半部分配对：对于 dim_size=4，
    // (dim0, dim2) 和 (dim1, dim3) 配对
    // 或者说是 (0, head_dim/2), (1, head_dim/2+1) 的配对方式

    // dim 2-3 配对：angle = 0.01
    // q[2] = 0*cos(0.01) - 0*sin(0.01) = 0
    EXPECT_NEAR(q[2], std::sin(1.0), 1e-6);
    EXPECT_NEAR(q[3], 0.0, 1e-6);

    // K 向量 [0, 1, 0, 0]：
    EXPECT_NEAR(k[0], 0.0, 1e-6);
    EXPECT_NEAR(k[1], std::cos(0.01), 1e-6);
    EXPECT_NEAR(k[2], 0.0, 1e-6);
    EXPECT_NEAR(k[3], std::sin(0.01), 1e-6);
}

// ---------------------------------------------------------------------------
// 测试：旋转是正交变换 —— 模长保持不变
//
// 旋转是正交变换：每个 head 的向量模长（L2 范数）必须保持不变。
// 这是 RoPE 的核心数学性质，确保 attention score 不会因为位置编码
// 而改变向量的大小。
// ---------------------------------------------------------------------------
TEST (rope_preserves_norm) {
    // 旋转是正交变换：每个 head 的向量模长必须保持不变。
    const int n_heads = 3, n_kv = 1, head_dim = 8, pos = 17;
    std::vector<float> q(n_heads * head_dim), k(n_kv * head_dim);

    // 计算旋转前的 L2 范数平方
    double norm_q = 0.0, norm_k = 0.0;
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = static_cast<float>(static_cast<int>(i * 7 % 11) - 5) * 0.25f;
        norm_q += q[i] * q[i];
    }
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = static_cast<float>(static_cast<int>(i * 5 % 7) - 3) * 0.5f;
        norm_k += k[i] * k[i];
    }

    // 执行 RoPE 旋转
    rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, pos, 10000.0f);

    // 计算旋转后的 L2 范数平方
    double after_q = 0.0, after_k = 0.0;
    for (float v: q) after_q += v * v;
    for (float v: k) after_k += v * v;

    // 旋转前后模长应该保持不变（正交变换）
    EXPECT_NEAR(after_q, norm_q, 1e-4);
    EXPECT_NEAR(after_k, norm_k, 1e-5);
}

// ---------------------------------------------------------------------------
// 测试：GQA 场景下每个 head 独立旋转
//
// 2 个 q head、1 个 kv head：每个 head 独立旋转，
// 且每个 head 的结果与单 head 情形一致。
// 这是 GQA 架构下 RoPE 的正确性保证。
// ---------------------------------------------------------------------------
TEST (rope_gqa_applies_to_all_heads) {
    // 2 个 q head、1 个 kv head：每个 head 独立旋转，
    // 且每个 head 的结果与单 head 情形一致。
    float q[8] = {1, 0, 0, 0, 0, 1, 0, 0}; // [head0: 1,0,0,0 | head1: 0,1,0,0]
    float k[4] = {1, 0, 0, 0};               // 唯一的 kv head

    rope_ref(q, k, 2, 1, 4, 1, 10000.0f);

    // head 0：dim 0-1 配对，angle = 1.0
    EXPECT_NEAR(q[0], std::cos(1.0), 1e-6);  // head 0, dim 0
    EXPECT_NEAR(q[2], std::sin(1.0), 1e-6);  // head 0, dim 2

    // head 1：dim 1-3 配对，angle = 0.01
    EXPECT_NEAR(q[5], std::cos(0.01), 1e-6); // head 1, dim 1 = index 5
    EXPECT_NEAR(q[7], std::sin(0.01), 1e-6); // head 1, dim 3 = index 7

    // KV head
    EXPECT_NEAR(k[0], std::cos(1.0), 1e-6);
}

// ---------------------------------------------------------------------------
// 测试：pos=0 时 RoPE 应为恒等变换
//
// 当位置为 0 时，所有旋转角度为 0（cos(0)=1, sin(0)=0），
// RoPE 应该是一个恒等变换，不改变输入向量。
// ---------------------------------------------------------------------------
TEST (rope_position_zero_is_identity) {
    // pos=0 时角度全为 0，RoPE 应为恒等变换。
    float q[6] = {1, 2, 3, 4, 5, 6}; // 3 heads × 2 dims
    float k[2] = {7, 8};              // 1 kv head × 2 dims

    // 保存原始值
    const float q0[6] = {1, 2, 3, 4, 5, 6};
    const float k0[2] = {7, 8};

    // 执行 RoPE（pos=0）
    rope_ref(q, k, 3, 1, 2, 0, 10000.0f);

    // 所有值应该不变（cos(0)=1, sin(0)=0，所以旋转矩阵是单位矩阵）
    for (int i = 0; i < 6; ++i) EXPECT_NEAR(q[i], q0[i], 1e-6);
    for (int i = 0; i < 2; ++i) EXPECT_NEAR(k[i], k0[i], 1e-6);
}