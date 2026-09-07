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

// ============================================================================
// 字节预算护栏
// ============================================================================
// 按槽数限界时，用户填一个大数会让 cache 悄悄吃掉数 GB 并把机器推进重度换页
// （实测 swap used 11.3 GB / 12 GB），此后所有计时不可信。字节预算让上限显式。

TEST(expert_store_budget_requires_register_first) {
    // 未 register 就设预算 → fail-fast（per_expert_bytes 未知，无法换算槽数）
    const std::string path = "test_expert_store_budget0.bin";
    const int bb = 64;
    std::vector<uint64_t> offs;
    make_test_file(path, 2, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    EXPECT_TRUE(!store.set_cache_budget(1024, &err));
    EXPECT_TRUE(!err.empty());
    std::remove(path.c_str());
}

TEST(expert_store_budget_rejects_smaller_than_one_expert) {
    // 预算装不下单个专家 → fail-fast。静默降级成 slots=0 会让用户误以为预算
    // 生效，而实际行为完全不同（每次访问都 pread）。
    const std::string path = "test_expert_store_budget1.bin";
    const int bb = 256;   // 单专家 = 3 * 256 = 768 B
    std::vector<uint64_t> offs;
    make_test_file(path, 2, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    store.register_expert(0, 0, offs[0], bb, offs[1], bb, offs[2], bb, 4, 4, 4);
    EXPECT_EQ(store.per_expert_bytes(), (uint64_t)(3 * bb));
    EXPECT_TRUE(!store.set_cache_budget(3 * bb - 1, &err));  // 差 1 字节也不放行
    EXPECT_TRUE(!err.empty());
    // 恰好等于单专家字节数 → 放行（slots=1）
    EXPECT_TRUE(store.set_cache_budget(3 * bb, &err));
    std::remove(path.c_str());
}

TEST(expert_store_budget_derives_slots_and_stays_within_budget) {
    // 预算换算槽数：budget / per_expert_bytes，且实际占用不超预算
    const std::string path = "test_expert_store_budget2.bin";
    const int n_experts = 6, bb = 128;   // 单专家 = 384 B
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);

    // 预算 1000 B → 1000/384 = 2 槽
    EXPECT_TRUE(store.set_cache_budget(1000, &err));
    EXPECT_EQ(store.cache_budget_bytes(), (uint64_t)1000);

    // 访问全部 6 个专家：只有 2 槽，必然发生淘汰，且占用不超预算
    for (int e = 0; e < n_experts; ++e) {
        auto w = store.get(0, e);
        uint8_t g = static_cast<uint8_t>((e * 10 + 0) & 0xFF);
        EXPECT_EQ(std::memcmp(w.gate, std::vector<uint8_t>(bb, g).data(), bb), 0);
    }
    EXPECT_TRUE(store.stats().evictions > 0);
    // 重访最早的专家应 miss（已被淘汰）
    const size_t m_before = store.stats().misses;
    store.get(0, 0);
    EXPECT_EQ(store.stats().misses, m_before + 1);
    std::remove(path.c_str());
}

TEST(expert_store_budget_data_correctness) {
    // 预算模式下 pread 数据仍须逐位正确（预算只改容量，不改语义）
    const std::string path = "test_expert_store_budget3.bin";
    const int n_experts = 4, bb = 200;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);
    EXPECT_TRUE(store.set_cache_budget(3 * bb * 2, &err));  // 2 槽
    for (int e = 0; e < n_experts; ++e) {
        auto w = store.get(0, e);
        for (int which = 0; which < 3; ++which) {
            const uint8_t *p = which == 0 ? w.gate : (which == 1 ? w.up : w.down);
            uint8_t fill = static_cast<uint8_t>((e * 10 + which) & 0xFF);
            EXPECT_EQ(std::memcmp(p, std::vector<uint8_t>(bb, fill).data(), bb), 0);
        }
    }
    std::remove(path.c_str());
}

// ============================================================================
// B-2 异步预取护栏（防死锁回归）
// ============================================================================
// 这两个测试锁的是实测踩到的两个死锁（AGENTS.md 坑 #27）：
//   ① 单 CV + notify_one 唤醒错误等待者 → 双方永久互等
//   ② 槽选择覆盖在飞槽 → 主线程重查自己的 key 得 nullptr → 永等
// 死锁的表现是测试挂死而非失败，所以用"多次循环 + 专家数 > 槽数"制造槽复用压力。

TEST(expert_store_prefetch_correctness) {
    const std::string path = "test_expert_store_pf1.bin";
    const int n_experts = 12, bb = 512;   // 专家数 > 预取槽数，强制槽复用
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);
    store.set_cache_slots(2);
    EXPECT_TRUE(store.prefetch_enable(4, &err));   // 4 槽 < 12 专家

    // 多轮循环：每轮先入队全部专家再逐个 get，制造"槽被复用给别的 key"的压力
    for (int round = 0; round < 5; ++round) {
        for (int e = 0; e < n_experts; ++e) store.prefetch_enqueue(0, e);
        for (int e = 0; e < n_experts; ++e) {
            auto w = store.get(0, e);
            for (int which = 0; which < 3; ++which) {
                const uint8_t *p = which == 0 ? w.gate : (which == 1 ? w.up : w.down);
                uint8_t fill = static_cast<uint8_t>((e * 10 + which) & 0xFF);
                EXPECT_EQ(std::memcmp(p, std::vector<uint8_t>(bb, fill).data(), bb), 0);
            }
        }
    }
    store.prefetch_disable();
    std::remove(path.c_str());
}

TEST(expert_store_prefetch_slots_lt_experts_no_hang) {
    // 极端压力：预取槽数远小于专家数，且反复入队同一批专家。
    // 若槽选择会覆盖在飞槽、或等待条件不容忍"槽消失"，这里会挂死。
    const std::string path = "test_expert_store_pf2.bin";
    const int n_experts = 16, bb = 256;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);
    store.set_cache_slots(0);                  // 全 miss，逼预取路径承担全部加载
    EXPECT_TRUE(store.prefetch_enable(2, &err));   // 只有 2 槽，16 个专家

    for (int round = 0; round < 10; ++round) {
        for (int e = 0; e < n_experts; ++e) store.prefetch_enqueue(0, e);
        for (int e = 0; e < n_experts; ++e) {
            auto w = store.get(0, e);
            uint8_t fill = static_cast<uint8_t>((e * 10 + 0) & 0xFF);
            EXPECT_EQ(std::memcmp(w.gate, std::vector<uint8_t>(bb, fill).data(), bb), 0);
        }
    }
    // 析构会自动 prefetch_disable；这里显式调一次验证可重复调用不崩
    store.prefetch_disable();
    store.prefetch_disable();
    std::remove(path.c_str());
}

TEST(expert_store_prefetch_fallback_without_enqueue) {
    // 完全不入队：get() 必须回退同步 LRU 路径且数据正确（不能因预取启用就挂死）
    const std::string path = "test_expert_store_pf3.bin";
    const int n_experts = 4, bb = 128;
    std::vector<uint64_t> offs;
    make_test_file(path, n_experts, bb, offs);
    tinyqwen::ExpertStore store;
    std::string err;
    EXPECT_TRUE(store.open(path, &err));
    for (int e = 0; e < n_experts; ++e)
        store.register_expert(0, e, offs[e*3], bb, offs[e*3+1], bb, offs[e*3+2], bb, 4, 4, 4);
    store.set_cache_slots(2);
    EXPECT_TRUE(store.prefetch_enable(4, &err));
    for (int e = 0; e < n_experts; ++e) {
        auto w = store.get(0, e);   // 从未 enqueue → 必走回退
        uint8_t fill = static_cast<uint8_t>((e * 10 + 0) & 0xFF);
        EXPECT_EQ(std::memcmp(w.gate, std::vector<uint8_t>(bb, fill).data(), bb), 0);
    }
    EXPECT_TRUE(store.stats().pf_fallbacks > 0);
    store.prefetch_disable();
    std::remove(path.c_str());
}
