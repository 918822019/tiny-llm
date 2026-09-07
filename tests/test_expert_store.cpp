// ============================================================================
// test_expert_store.cpp — MoE 专家 SSD 卸载存储（ExpertStore）单元测试
// ============================================================================
// 验证：
//   1. get() 返回的数据 == 文件中该专家区域的内容（pread 正确）
//   2. LRU 命中/淘汰：slots=2 时访问 3 个专家，第 3 个淘汰 LRU，重访被淘汰者算 miss
//   3. slots=0（全 miss）仍正确返回数据
//   4. stats 计数（hits/misses/evictions/bytes_read）
//
// 测试构造一个临时二进制文件，按已知偏移写若干"专家块"（每块填可识别字节模式），
// 不依赖真实 .tqwen 格式——ExpertStore 只按 offset+nbytes pread。
// ============================================================================

#include "test_framework.h"

#include "expert_store.h"

#include <cstdio>     // std::remove
#include <cstring>    // std::memcmp
#include <fstream>
#include <string>
#include <vector>

namespace {

// 构造测试文件：3 个专家，每个 3 块（gate/up/down），每块 block_bytes 字节。
// 字节模式：block[layer][expert][which] = (layer*100 + expert*10 + which) & 0xFF
std::string make_test_file(const std::string &path, int n_experts, int block_bytes,
                            std::vector<uint64_t> &offsets) {
    // 布局：每个专家的 3 块连续排列，专家之间隔开
    std::vector<uint8_t> buf;
    buf.reserve(static_cast<size_t>(n_experts) * 3 * block_bytes);
    for (int e = 0; e < n_experts; ++e) {
        for (int which = 0; which < 3; ++which) {
            uint8_t fill = static_cast<uint8_t>((e * 10 + which) & 0xFF);
            for (int i = 0; i < block_bytes; ++i) buf.push_back(fill);
        }
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char *>(buf.data()), buf.size());
    f.close();
    // offsets[e][which] = e * 3 * block_bytes + which * block_bytes
    offsets.resize(static_cast<size_t>(n_experts) * 3);
    for (int e = 0; e < n_experts; ++e)
        for (int w = 0; w < 3; ++w)
            offsets[static_cast<size_t>(e) * 3 + w] =
                static_cast<uint64_t>(e) * 3 * block_bytes + w * block_bytes;
    return path;
}

} // namespace

TEST(expert_store_pread_correctness) {
    const std::string path = "test_expert_store_tmp.bin";
    const int n_experts = 4, bb = 256;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);

    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    store.set_cache_slots(4);  // 全能装下
    for (int e = 0; e < n_experts; ++e) {
        store.register_expert(/*layer*/ 0, e,
            offs[e * 3 + 0], bb, offs[e * 3 + 1], bb, offs[e * 3 + 2], bb,
            /*inter*/ 16, /*hidden*/ 32, /*group*/ 8);
    }
    for (int e = 0; e < n_experts; ++e) {
        auto w = store.get(0, e);
        uint8_t g = static_cast<uint8_t>((e * 10 + 0) & 0xFF);
        uint8_t u = static_cast<uint8_t>((e * 10 + 1) & 0xFF);
        uint8_t d = static_cast<uint8_t>((e * 10 + 2) & 0xFF);
        EXPECT_EQ(std::memcmp(w.gate, std::vector<uint8_t>(bb, g).data(), bb), 0);
        EXPECT_EQ(std::memcmp(w.up,   std::vector<uint8_t>(bb, u).data(), bb), 0);
        EXPECT_EQ(std::memcmp(w.down, std::vector<uint8_t>(bb, d).data(), bb), 0);
    }
    // 第一次全 miss
    EXPECT_EQ(store.stats().misses, (size_t)n_experts);
    EXPECT_EQ(store.stats().hits, (size_t)0);
    std::remove(path.c_str());
}

TEST(expert_store_lru_eviction) {
    const std::string path = "test_expert_store_lru.bin";
    const int n_experts = 3, bb = 128;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);

    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    store.set_cache_slots(2);  // 只能装 2 个，第 3 个淘汰 LRU
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 8, 8, 4);

    store.get(0, 0);  // miss, slots=[0]
    store.get(0, 1);  // miss, slots=[0,1]
    store.get(0, 2);  // miss, evict LRU(0), slots=[1,2]
    EXPECT_EQ(store.stats().misses, (size_t)3);
    EXPECT_EQ(store.stats().evictions, (size_t)1);
    store.get(0, 1);  // hit (1 还在)
    EXPECT_EQ(store.stats().hits, (size_t)1);
    store.get(0, 0);  // miss (0 被淘汰了)
    EXPECT_EQ(store.stats().misses, (size_t)4);
    std::remove(path.c_str());
}

TEST(expert_store_zero_slots_all_miss) {
    const std::string path = "test_expert_store_zero.bin";
    const int n_experts = 2, bb = 64;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    store.set_cache_slots(0);  // 全 miss，每次现读
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);
    for (int e = 0; e < n_experts; ++e) {
        auto w = store.get(0, e);
        uint8_t g = static_cast<uint8_t>((e * 10 + 0) & 0xFF);
        EXPECT_EQ(std::memcmp(w.gate, std::vector<uint8_t>(bb, g).data(), bb), 0);
    }
    EXPECT_EQ(store.stats().misses, (size_t)n_experts);
    EXPECT_EQ(store.stats().hits, (size_t)0);
    EXPECT_EQ(store.stats().bytes_read, (size_t)(n_experts * 3 * bb));
    std::remove(path.c_str());
}
