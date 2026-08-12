#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

TEST (attention_single_position) {
    // 只有一个缓存位置：单元素 softmax 恒为 1，所以 out == v0。
    const float q[2] = {1.0f, 0.0f};
    const float k[2] = {2.0f, 0.0f}; // 布局 [kv=1][max=2][hd=2]，只用 pos 0
    const float v[2 * 2] = {3.0f, 4.0f, 99.0f, 99.0f};
    float out[2];
    attention_decode_ref(q, k, v, /*seq_len=*/1, /*max_seq_len=*/2, 1, 1, 2, 0.5f, out);
    EXPECT_NEAR(out[0], 3.0, 1e-6);
    EXPECT_NEAR(out[1], 4.0, 1e-6);
}

TEST (attention_two_positions_manual) {
    // q=[1,0]，K=[[1,0],[0,1]]，V=[[1,0],[0,1]]，scale=1
    // scores=[1,0] -> softmax=[e/(e+1), 1/(e+1)]
    const float q[2] = {1.0f, 0.0f};
    const float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float out[2];
    attention_decode_ref(q, k, v, 2, 2, 1, 1, 2, 1.0f, out);
    const double e = std::exp(1.0);
    EXPECT_NEAR(out[0], e / (e + 1.0), 1e-6);
    EXPECT_NEAR(out[1], 1.0 / (e + 1.0), 1e-6);
}

TEST (attention_gqa_shares_kv_head) {
    // 2 个 q head、1 个 kv head：两个 head 都 attend 到同一份 K/V。
    const float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float k[2] = {1.0f, 0.0f};
    const float v[2] = {5.0f, 6.0f};
    float out[4];
    attention_decode_ref(q, k, v, 1, 1, 2, 1, 2, 0.125f, out);
    EXPECT_NEAR(out[0], 5.0, 1e-6);
    EXPECT_NEAR(out[1], 6.0, 1e-6);
    EXPECT_NEAR(out[2], 5.0, 1e-6);
    EXPECT_NEAR(out[3], 6.0, 1e-6);
}

TEST (attention_respects_max_seq_len_stride) {
    // 未使用槽位填上垃圾值，且绝不能被读到：专门用来抓
    // [n_kv_heads][max_seq_len][head_dim] 与 seq_len 之间的 stride bug。
    const int max_seq = 4, seq_len = 2, head_dim = 2;
    std::vector<float> k(max_seq * head_dim, 999.0f);
    std::vector<float> v(max_seq * head_dim, 999.0f);
    // pos0: K=[1,0] V=[1,0]；pos1: K=[0,1] V=[0,1]
    k[0] = 1;
    k[1] = 0;
    v[0] = 1;
    v[1] = 0;
    k[2] = 0;
    k[3] = 1;
    v[2] = 0;
    v[3] = 1;

    const float q[2] = {1.0f, 0.0f};
    float out[2];
    attention_decode_ref(q, k.data(), v.data(), seq_len, max_seq, 1, 1, head_dim, 1.0f, out);
    const double e = std::exp(1.0);
    EXPECT_NEAR(out[0], e / (e + 1.0), 1e-6);
    EXPECT_NEAR(out[1], 1.0 / (e + 1.0), 1e-6);
}

TEST (attention_matches_softmax_kernel) {
    // 用小规模伪随机数据，和"显式 score + softmax_ref"的独立实现交叉验证。
    const int n_heads = 2, n_kv = 1, head_dim = 4, seq_len = 3, max_seq = 3;
    const float scale = 0.5f;
    std::vector<float> q(n_heads * head_dim), k(n_kv * max_seq * head_dim),
            v(n_kv * max_seq * head_dim), out(n_heads * head_dim);
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = static_cast<float>(static_cast<int>(i * 11 % 7) - 3) * 0.25f;
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = static_cast<float>(static_cast<int>(i * 5 % 9) - 4) * 0.25f;
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(static_cast<int>(i * 7 % 5) - 2) * 0.5f;

    attention_decode_ref(q.data(), k.data(), v.data(), seq_len, max_seq, n_heads, n_kv,
                         head_dim, scale, out.data());

    const int heads_per_kv = n_heads / n_kv;
    for (int h = 0; h < n_heads; ++h) {
        const int kv = h / heads_per_kv;
        std::vector<float> scores(seq_len);
        for (int t = 0; t < seq_len; ++t) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += (double) q[h * head_dim + d] * k[(kv * max_seq + t) * head_dim + d];
            }
            scores[t] = (float) dot * scale;
        }
        std::vector<float> p(seq_len);
        softmax_ref(scores.data(), p.data(), seq_len);
        for (int d = 0; d < head_dim; ++d) {
            double expect = 0.0;
            for (int t = 0; t < seq_len; ++t) {
                expect += (double) p[t] * v[(kv * max_seq + t) * head_dim + d];
            }
            EXPECT_NEAR(out[h * head_dim + d], expect, 1e-5);
        }
    }
}
