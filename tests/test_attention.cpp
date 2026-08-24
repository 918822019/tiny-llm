// ============================================================================
// test_attention.cpp — 注意力机制（Attention）参考实现单元测试
// ============================================================================
// 本文件测试 attention_decode_ref 参考实现，覆盖以下场景：
//   1. 单个缓存位置（单元素 softmax 恒为 1）
//   2. 两个位置的手工计算值验证
//   3. GQA（分组查询注意力）KV 头共享验证
//   4. max_seq_len stride 正确性验证（未使用槽位不被读到）
//   5. 与独立 softmax 实现交叉验证
// 测试使用的 attention_decode_ref 定义在 ref_ops.h 中，是纯 C++ 参考实现，
// 用于验证后续各种优化实现（NEON、CUDA 等）的数值正确性。
// ============================================================================

#include "test_framework.h"

#include <cmath>
#include <vector>

#include "ref_ops.h"

using namespace tinyqwen;

// =============================================================================
// attention_single_position — 单个缓存位置测试
// =============================================================================
// 当 KV 缓存中只有一个位置时，softmax 对单元素输入恒输出 1.0，
// 因此 attention 输出应当等于 V 向量本身。
// 参数说明：
//   q — 查询向量 [1.0, 0.0]
//   k — 键向量 [2.0, 0.0]，布局 [kv_heads=1][max_seq=2][head_dim=2]
//   v — 值向量 [3.0, 4.0, 99.0, 99.0]，第二个位置填充垃圾值
//   seq_len=1：只使用第一个位置
//   scale=0.5：缩放因子
// 预期输出：[3.0, 4.0]
TEST (attention_single_position) {
    // 只有一个缓存位置：单元素 softmax 恒为 1，所以 out == v0。
    const float q[2] = {1.0f, 0.0f};
    const float k[2] = {2.0f, 0.0f}; // 布局 [kv=1][max=2][hd=2]，只用 pos 0
    const float v[2 * 2] = {3.0f, 4.0f, 99.0f, 99.0f};
    float out[2];
    // 调用参考实现：n_heads=1, n_kv=1, head_dim=2, scale=0.5
    attention_decode_ref(q, k, v, /*seq_len=*/1, /*max_seq_len=*/2, 1, 1, 2, 0.5f, out);
    EXPECT_NEAR(out[0], 3.0, 1e-6);
    EXPECT_NEAR(out[1], 4.0, 1e-6);
}

// =============================================================================
// attention_two_positions_manual — 两个位置手工计算验证
// =============================================================================
// 构造简化的 q、K、V，使得 attention score 可以直接手算验证。
//   q = [1, 0]
//   K = [[1, 0], [0, 1]]  即 K[0] = [1,0], K[1] = [0,1]
//   V = [[1, 0], [0, 1]]  即 V[0] = [1,0], V[1] = [0,1]
//   scale = 1.0
// 计算过程：
//   scores = q @ K^T = [1*1+0*0, 1*0+0*1] = [1, 0]
//   softmax = [e/(e+1), 1/(e+1)]
//   out = softmax[0]*V[0] + softmax[1]*V[1]
// 预期输出：
//   out[0] = e/(e+1) * 1 + 1/(e+1) * 0 = e/(e+1)
//   out[1] = e/(e+1) * 0 + 1/(e+1) * 1 = 1/(e+1)
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

// =============================================================================
// attention_gqa_shares_kv_head — GQA（分组查询注意力）KV 头共享测试
// =============================================================================
// 验证当 q heads 数量大于 kv heads 数量时，多个 q head 共享同一份 K/V 数据。
// 配置：2 个 q head、1 个 kv head，两个 head 都 attend 到同一份 K/V。
//   q_head0 = [1, 0], q_head1 = [0, 1]
//   K = [1, 0], V = [5, 6]
//   scale = 0.125
// 预期输出：两个 head 的输出相同，均为 [5, 6]
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

// =============================================================================
// attention_respects_max_seq_len_stride — max_seq_len stride 正确性测试
// =============================================================================
// 验证 attention 实现正确地使用 max_seq_len 作为 stride 来访问 K/V 缓存，
// 不会错误地读到 seq_len 和 max_seq_len 之间的垃圾数据。
// 构造 max_seq=4、seq_len=2 的 K/V 缓存，未使用位置（pos 2,3）填入 999.0
// 作为哨兵值。如果实现错误地把 stride 当成了 seq_len，就会读到这些哨兵值
// 导致输出偏差异常大。
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

// =============================================================================
// attention_matches_softmax_kernel — 与独立 softmax 实现交叉验证
// =============================================================================
// 使用伪随机数据，将 attention_decode_ref 的输出与"显式计算 score +
// 独立调用 softmax_ref + 加权求和"的结果进行交叉验证。
// 配置：2 个 q head、1 个 kv head（GQA）、head_dim=4、seq_len=3
// 测试方法：
//   1. 调用 attention_decode_ref 得到输出
//   2. 对每个 head 独立计算 score（q @ k），调用 softmax_ref 得到概率分布
//   3. 用概率分布加权求和 V，得到期望输出
//   4. 比较两者是否一致
TEST (attention_matches_softmax_kernel) {
    // 用小规模伪随机数据，和"显式 score + softmax_ref"的独立实现交叉验证。
    const int n_heads = 2, n_kv = 1, head_dim = 4, seq_len = 3, max_seq = 3;
    const float scale = 0.5f;
    std::vector<float> q(n_heads * head_dim), k(n_kv * max_seq * head_dim),
            v(n_kv * max_seq * head_dim), out(n_heads * head_dim);
    // 使用确定性伪随机序列填充 q、k、v
    for (size_t i = 0; i < q.size(); ++i)
        q[i] = static_cast<float>(static_cast<int>(i * 11 % 7) - 3) * 0.25f;
    for (size_t i = 0; i < k.size(); ++i)
        k[i] = static_cast<float>(static_cast<int>(i * 5 % 9) - 4) * 0.25f;
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = static_cast<float>(static_cast<int>(i * 7 % 5) - 2) * 0.5f;

    // 调用参考 attention 实现
    attention_decode_ref(q.data(), k.data(), v.data(), seq_len, max_seq, n_heads, n_kv,
                         head_dim, scale, out.data());

    // 计算每个 q head 对应的 kv head 索引，GQA 中 heads_per_kv = n_heads / n_kv
    const int heads_per_kv = n_heads / n_kv;
    for (int h = 0; h < n_heads; ++h) {
        const int kv = h / heads_per_kv; // 该 q head 对应的 kv head 索引
        std::vector<float> scores(seq_len);
        // 计算每个位置的 attention score：q[h] @ k[kv][t] * scale
        for (int t = 0; t < seq_len; ++t) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += (double) q[h * head_dim + d] * k[(kv * max_seq + t) * head_dim + d];
            }
            scores[t] = (float) dot * scale;
        }
        // 对 scores 做 softmax 得到概率分布
        std::vector<float> p(seq_len);
        softmax_ref(scores.data(), p.data(), seq_len);
        // 用概率分布加权求和 V，得到该 head 的期望输出
        for (int d = 0; d < head_dim; ++d) {
            double expect = 0.0;
            for (int t = 0; t < seq_len; ++t) {
                expect += (double) p[t] * v[(kv * max_seq + t) * head_dim + d];
            }
            EXPECT_NEAR(out[h * head_dim + d], expect, 1e-5);
        }
    }
}
// ===========================================================================
// fp16-KV 融合 attention 测试（仅 aarch64）
// ===========================================================================
// 验证 attention_decode_f16kv_neon（读 fp16 K/V、寄存器内转 fp32）与
// "先把 fp16 反量化成 fp32 再走 attention_decode_ref"数值一致。
// 两者用同一份 fp16 K/V，唯一差异是 NEON float 累加 vs ref double 累加，
// 故按 1e-4 容差对齐。这证明融合 kernel 计算正确（fp16 压缩损失是共享的）。
#if defined(__aarch64__) || defined(_M_ARM64)
TEST(attention_f16kv_fused_matches_dequant_ref) {
    const int seq_len = 5, max_seq = 8, n_heads = 2, n_kv = 1, head_dim = 16;
    // 构造 fp32 的 q / k / v
    std::vector<float> q(n_heads * head_dim);
    std::vector<float> k(static_cast<size_t>(n_kv) * max_seq * head_dim);
    std::vector<float> v(static_cast<size_t>(n_kv) * max_seq * head_dim);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * ((i * 7) % 13) - 0.5f;
    for (size_t i = 0; i < k.size(); ++i) k[i] = 0.1f * ((i * 5) % 11) - 0.4f;
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.1f * ((i * 3) % 9) - 0.3f;
    // 把 k/v 转成 fp16（模拟 fp16 KV cache），再反量化回 fp32 作为参考输入
    std::vector<uint16_t> k_f16(k.size()), v_f16(v.size());
    std::vector<float> k_deq(k.size()), v_deq(v.size());
    for (size_t i = 0; i < k.size(); ++i) {
        k_f16[i] = tinyqwen::float_to_half(k[i]);
        k_deq[i] = tinyqwen::half_to_float(k_f16[i]);
    }
    for (size_t i = 0; i < v.size(); ++i) {
        v_f16[i] = tinyqwen::float_to_half(v[i]);
        v_deq[i] = tinyqwen::half_to_float(v_f16[i]);
    }
    const float scale = 0.25f;
    // 融合 kernel：直接读 fp16
    std::vector<float> out_fused(n_heads * head_dim);
    tinyqwen::attention_decode_f16kv_neon(q.data(), k_f16.data(), v_f16.data(), seq_len,
                                          max_seq, n_heads, n_kv, head_dim, scale,
                                          out_fused.data());
    // 参考：反量化成 fp32 后走 ref
    std::vector<float> out_ref(n_heads * head_dim);
    tinyqwen::attention_decode_ref(q.data(), k_deq.data(), v_deq.data(), seq_len, max_seq,
                                   n_heads, n_kv, head_dim, scale, out_ref.data());
    for (int i = 0; i < n_heads * head_dim; ++i) {
        EXPECT_NEAR(out_fused[i], out_ref[i], 1e-4);
    }
}
#endif
