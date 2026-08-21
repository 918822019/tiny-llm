// ============================================================================
// test_kv_cache.cpp — KV 缓存（KvCache）单元测试
// ============================================================================
// 本文件测试 KvCache 类的核心功能，包括：
//   1. 缓存布局和内存大小计算的正确性验证
//   2. K/V 追加（append）和读取（read-back）的完整流程测试
//   3. K 和 V 是两块独立内存的验证（写一边不影响另一边）
//   4. reset() 重置 seq_len 的功能验证
//
// KvCache 是自回归生成流程中的核心数据结构：每个 token 生成后，
// 当前层的 K/V 投影结果被追加到缓存中，后续 token 的 attention 计算
// 会读取缓存中所有历史位置的 K/V 数据。
// 缓存的布局为：
//   对于 K 缓存：float[kv_heads][max_seq_len][head_dim]
//   对于 V 缓存：float[kv_heads][max_seq_len][head_dim]
// 索引公式为：(head_idx * max_seq_len + position) * head_dim
// ============================================================================

#include "test_framework.h"

#include <cstring>
#include <vector>

#include "kv_cache.h"

using namespace tinyqwen;

// =============================================================================
// kv_cache_layout_and_append — 缓存布局和追加操作测试
// =============================================================================
// 测试 KvCache 的完整生命周期：
//   1. 创建缓存：2 层、2 个 KV 头、最大序列长度 4、head_dim=3
//   2. 验证初始 seq_len 为 0
//   3. 验证内存总大小为 2 * n_layers * n_kv * max_seq * head_dim * sizeof(float)
//      （系数 2 是因为 K 和 V 各有一份）
//   4. 向所有层/头的 pos=0 位置写入数据，使用与模型相同的索引公式
//   5. advance(1) 后 seq_len 变为 1
//   6. 读回验证数据正确性
//   7. reset() 后 seq_len 回到 0
TEST (kv_cache_layout_and_append) {
    const int n_layers = 2, n_kv = 2, max_seq = 4, head_dim = 3;
    KvCache cache(n_layers, n_kv, max_seq, head_dim);
    // 初始 seq_len 为 0
    EXPECT_EQ(cache.seq_len(), 0);
    // 内存大小验证：2 份（K+V）* 2 层 * 2 头 * 4 位置 * 3 维度 * 4 字节 = 384 字节
    EXPECT_EQ(cache.memory_bytes(),
              (size_t) 2 * n_layers * n_kv * max_seq * head_dim * sizeof(float));

    // 用和模型完全相同的索引公式 (h * max_seq + pos) * head_dim
    // 向所有层/头追加一个位置，再读回验证。
    const int pos = 0;
    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < n_kv; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                // K 写入正值：100 + 层偏移 + 头偏移 + 维度偏移
                cache.k(l)[(h * max_seq + pos) * head_dim + d] = 100.0f + l * 10 + h + d * 0.1f;
                // V 写入负值：-(100 + 层偏移 + 头偏移 + 维度偏移)
                cache.v(l)[(h * max_seq + pos) * head_dim + d] = -(100.0f + l * 10 + h + d * 0.1f);
            }
        }
    }
    cache.advance(1); // 推进序列长度
    EXPECT_EQ(cache.seq_len(), 1);

    // 读回验证：K 为正、V 为负，数值正确
    for (int l = 0; l < n_layers; ++l) {
        for (int h = 0; h < n_kv; ++h) {
            EXPECT_NEAR(cache.k(l)[(h * max_seq) * head_dim + 1], 100.0f + l * 10 + h + 0.1f, 1e-6);
            EXPECT_NEAR(cache.v(l)[(h * max_seq) * head_dim + 2], -(100.0f + l * 10 + h + 0.2f),
                        1e-6);
        }
    }

    // 重置缓存，seq_len 应回到 0
    cache.reset();
    EXPECT_EQ(cache.seq_len(), 0);
}

// =============================================================================
// kv_cache_k_v_independent — K/V 独立内存测试
// =============================================================================
// 验证 K 和 V 缓存使用了不同的内存块，写入 K 不会影响 V 的读取。
// 这是一个基本的正确性检查：如果 K 和 V 指向了同一块内存，会导致
// attention 计算完全错误。
TEST (kv_cache_k_v_independent) {
    // K、V 是两块独立内存，写一边不影响另一边。
    KvCache cache(1, 1, 2, 2);
    cache.k(0)[0] = 1.0f;  // 写入 K
    cache.v(0)[0] = 2.0f;  // 写入 V
    // 读回验证：K 和 V 各自保持自己的值
    EXPECT_NEAR(cache.k(0)[0], 1.0, 0.0);
    EXPECT_NEAR(cache.v(0)[0], 2.0, 0.0);
    // 验证 K 和 V 的指针确实不同（独立内存）
    EXPECT_TRUE(cache.k(0) != cache.v(0));
}