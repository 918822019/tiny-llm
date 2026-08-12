#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

TEST(rope_single_head_pos1) {
  // head_dim=4、pos=1、theta=10000 时 inv_freq = [1.0, 0.01]，可手算验证。
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float k[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  rope_ref(q, k, 1, 1, 4, 1, 10000.0f);

  EXPECT_NEAR(q[0], std::cos(1.0), 1e-6);
  EXPECT_NEAR(q[1], 0.0, 1e-6);
  EXPECT_NEAR(q[2], std::sin(1.0), 1e-6);
  EXPECT_NEAR(q[3], 0.0, 1e-6);

  EXPECT_NEAR(k[0], 0.0, 1e-6);
  EXPECT_NEAR(k[1], std::cos(0.01), 1e-6);
  EXPECT_NEAR(k[2], 0.0, 1e-6);
  EXPECT_NEAR(k[3], std::sin(0.01), 1e-6);
}

TEST(rope_preserves_norm) {
  // 旋转是正交变换：每个 head 的向量模长必须保持不变。
  const int n_heads = 3, n_kv = 1, head_dim = 8, pos = 17;
  std::vector<float> q(n_heads * head_dim), k(n_kv * head_dim);
  double norm_q = 0.0, norm_k = 0.0;
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = static_cast<float>(static_cast<int>(i * 7 % 11) - 5) * 0.25f;
    norm_q += q[i] * q[i];
  }
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = static_cast<float>(static_cast<int>(i * 5 % 7) - 3) * 0.5f;
    norm_k += k[i] * k[i];
  }

  rope_ref(q.data(), k.data(), n_heads, n_kv, head_dim, pos, 10000.0f);

  double after_q = 0.0, after_k = 0.0;
  for (float v : q) after_q += v * v;
  for (float v : k) after_k += v * v;
  EXPECT_NEAR(after_q, norm_q, 1e-4);
  EXPECT_NEAR(after_k, norm_k, 1e-5);
}

TEST(rope_gqa_applies_to_all_heads) {
  // 2 个 q head、1 个 kv head：每个 head 独立旋转，
  // 且每个 head 的结果与单 head 情形一致。
  float q[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  float k[4] = {1, 0, 0, 0};
  rope_ref(q, k, 2, 1, 4, 1, 10000.0f);

  EXPECT_NEAR(q[0], std::cos(1.0), 1e-6);          // head 0
  EXPECT_NEAR(q[2], std::sin(1.0), 1e-6);
  EXPECT_NEAR(q[5], std::cos(0.01), 1e-6);         // head 1：配对 0 的 x1 分量
  EXPECT_NEAR(q[7], std::sin(0.01), 1e-6);
  EXPECT_NEAR(k[0], std::cos(1.0), 1e-6);
}

TEST(rope_position_zero_is_identity) {
  // pos=0 时角度全为 0，RoPE 应为恒等变换。
  float q[6] = {1, 2, 3, 4, 5, 6};
  float k[2] = {7, 8};
  const float q0[6] = {1, 2, 3, 4, 5, 6};
  const float k0[2] = {7, 8};
  rope_ref(q, k, 3, 1, 2, 0, 10000.0f);
  for (int i = 0; i < 6; ++i) EXPECT_NEAR(q[i], q0[i], 1e-6);
  for (int i = 0; i < 2; ++i) EXPECT_NEAR(k[i], k0[i], 1e-6);
}
