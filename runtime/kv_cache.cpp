// ============================================================================
// kv_cache.cpp — KV Cache 的内存管理
// ============================================================================
// 本文件实现 KvCache 类，管理 Transformer 模型推理中的 Key-Value 缓存。
//
// 在自回归推理中，每个 token 的 attention 计算需要访问所有之前 token 的
// Key 和 Value。为了避免每步都重新计算，我们将已计算过的 K 和 V 缓存起来。
// 这就是 KV Cache 的核心思想——用空间换时间。
//
// 设计要点：
//   1. 一次申请一整块连续内存（arena），而不是每层各申请一次。
//      好处：只有一次分配，简单且快；连续内存对缓存友好；
//      将来要换成量化/mmap 时，换一整块比换几十小块容易。
//   2. 这块 arena 内部切成两大块：前半是 K，后半是 V。
//   3. 每块再按层切成 n_layers 个"平面"（plane），每个平面是
//      [n_kv_heads][max_seq_len][head_dim] 的三维数组。
//   4. 追加新 token 的写法：往每一层的平面上、第 seq_len 个槽位写数据，
//      全部写完后调 advance() 把 seq_len 提交 +1。
//
// 内存布局（arena 整体）：
//   [Layer 0 K plane][Layer 1 K plane]...[Layer N-1 K plane]
//   [Layer 0 V plane][Layer 1 V plane]...[Layer N-1 V plane]
// 每个 plane 的大小 = n_kv_heads * max_seq_len * head_dim 个 float
// ============================================================================

#include "kv_cache.h"

#include <cstdio>   // 标准输入输出（fprintf, stderr）
#include <cstdlib>  // 标准库（abort）

namespace tinyqwen {
    // =========================================================================
    // KvCache::KvCache() — 构造函数
    // =========================================================================
    // 参数：
    //   n_layers     — 需要缓存的层数（Qwen3.5 下只含 full attention 层）
    //   n_kv_heads   — KV 头数量（GQA 中可小于 Q 头数）
    //   max_seq_len  — 最大序列长度（KV cache 的容量上限）
    //   head_dim     — 每个注意力头的维度
    KvCache::KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim) {
        init(n_layers, n_kv_heads, max_seq_len, head_dim);
    }

    // =========================================================================
    // KvCache::init() — 初始化（或重新初始化）KV Cache
    // =========================================================================
    // 参数：
    //   n_layers, n_kv_heads, max_seq_len, head_dim — 各维度参数
    // 说明：计算 arena 总大小，一次性分配并清零。所有参数必须为正数，否则 abort。
    //       计算方式：
    //         layer_stride = n_kv_heads * max_seq_len * head_dim  （每层的平面大小）
    //         total = 2 * n_layers * layer_stride                  （K 和 V 各一份）
    //       清零的好处：万一索引写错读到没写过的槽位，看到的是 0 而不是随机
    //       脏数据，更容易发现问题。
    void KvCache::init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim) {
        // 参数合法性检查：所有维度必须为正
        if (n_layers <= 0 || n_kv_heads <= 0 || max_seq_len <= 0 || head_dim <= 0) {
            std::fprintf(stderr, "tinyqwen: KvCache::init with invalid dims\n");
            std::abort();
        }
        n_layers_ = n_layers;
        n_kv_heads_ = n_kv_heads;
        max_seq_len_ = max_seq_len;
        head_dim_ = head_dim;
        seq_len_ = 0; // 初始序列长度为 0
        // 一层的"平面"有多少个 float：
        // n_kv_heads 个头 * max_seq_len 个位置 * head_dim 维
        layer_stride_ = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
        // 乘 2 是 K、V 两块；全部清零
        data_.assign(2 * static_cast<size_t>(n_layers) * layer_stride_, 0.0f);
    }

    // =========================================================================
    // KvCache::reset() — 清空缓存（重置序列长度）
    // =========================================================================
    // 说明：只将 seq_len_ 置零，不擦除数据。下次写入新 token 时会覆盖旧槽位。
    //       这是最轻量的"清空"方式——只需修改一个计数器。
    void KvCache::reset() { seq_len_ = 0; }

    // =========================================================================
    // KvCache::k() — 获取指定层 K 缓存的起始地址
    // =========================================================================
    // 参数：
    //   layer — 层索引（0 到 n_layers_-1）
    // 返回值：该层 K 平面第一个元素的指针
    // 说明：K 的 n_layers 个平面排在 arena 前半段，层 l 的起点 = l * layer_stride_。
    float *KvCache::k(int layer) { return data_.data() + static_cast<size_t>(layer) * layer_stride_; }

    // =========================================================================
    // KvCache::v() — 获取指定层 V 缓存的起始地址
    // =========================================================================
    // 参数：
    //   layer — 层索引（0 到 n_layers_-1）
    // 返回值：该层 V 平面第一个元素的指针
    // 说明：V 的 n_layers 个平面排在 arena 后半段，所以层 l 的起点要加
    //       n_layers_ 个平面的偏移量。
    float *KvCache::v(int layer) {
        return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }

    // =========================================================================
    // KvCache::k() const — 获取指定层 K 缓存的起始地址（只读版本）
    // =========================================================================
    const float *KvCache::k(int layer) const {
        return data_.data() + static_cast<size_t>(layer) * layer_stride_;
    }

    // =========================================================================
    // KvCache::v() const — 获取指定层 V 缓存的起始地址（只读版本）
    // =========================================================================
    const float *KvCache::v(int layer) const {
        return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }

    // =========================================================================
    // KvCache::advance() — 提交新增的 token 到缓存
    // =========================================================================
    // 参数：
    //   n — 新增的 token 数量（通常为 1，批量 prefill 时为 prompt 长度）
    // 说明：将 seq_len_ 增加 n。如果超出 max_seq_len_ 容量上限，abort。
    //       调用方在 advance 之前负责将新 token 的 K/V 数据写入平面上
    //       第 seq_len_ 个槽位（即 pos = seq_len_ 的位置）。
    void KvCache::advance(int n) {
        // 容量检查：当前序列长度 + 新增数量不能超过最大容量
        if (seq_len_ + n > max_seq_len_) {
            std::fprintf(stderr, "tinyqwen: KV cache overflow: seq_len %d + %d > max %d\n", seq_len_,
                         n, max_seq_len_);
            std::abort(); // 溢出则终止，防止越界写
        }
        seq_len_ += n; // 递增序列长度
    }
} // namespace tinyqwen