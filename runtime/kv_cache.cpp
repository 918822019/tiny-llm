// KV cache 的内存管理。
//
// 设计要点：一次申请一整块连续内存（叫 arena），而不是每层各申请一次。
// 好处：
//   1. 只有一次分配，简单且快；
//   2. 连续内存对缓存友好；
//   3. 将来要换成量化/mmap 时，换一整块比换几十小块容易。
//
// 这块 arena 内部切成两大块：前半是 K，后半是 V；每块再按层切成 n_layers 个
// "平面"，每个平面是 [n_kv_heads][max_seq_len][head_dim]。
//
// 追加新 token 的写法：往每一层的平面上、第 seq_len 个槽位写数据，
// 全部写完后调 advance() 把 seq_len 提交 +1。

#include "kv_cache.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

KvCache::KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim) {
  init(n_layers, n_kv_heads, max_seq_len, head_dim);
}

void KvCache::init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim) {
  if (n_layers <= 0 || n_kv_heads <= 0 || max_seq_len <= 0 || head_dim <= 0) {
    std::fprintf(stderr, "tinyqwen: KvCache::init with invalid dims\n");
    std::abort();
  }
  n_layers_ = n_layers;
  n_kv_heads_ = n_kv_heads;
  max_seq_len_ = max_seq_len;
  head_dim_ = head_dim;
  seq_len_ = 0;
  // 一层的"平面"有多少个 float：n_kv_heads 个头 * max_seq_len 个位置 * head_dim。
  layer_stride_ = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
  // 乘 2 是 K、V 两块；全部清零。清零的好处：万一索引写错读到没写过的槽位，
  // 看到的是 0 而不是随机脏数据，更容易发现问题。
  data_.assign(2 * static_cast<size_t>(n_layers) * layer_stride_, 0.0f);
}

void KvCache::reset() { seq_len_ = 0; }

// K 的 n_layers 个平面排在 arena 前半段：层 l 的起点 = l * layer_stride_。
// V 的 n_layers 个平面排在前半段之后，所以层 l 的起点要再加 n_layers_ 个平面。
float* KvCache::k(int layer) { return data_.data() + static_cast<size_t>(layer) * layer_stride_; }
float* KvCache::v(int layer) {
  return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
}
const float* KvCache::k(int layer) const {
  return data_.data() + static_cast<size_t>(layer) * layer_stride_;
}
const float* KvCache::v(int layer) const {
  return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
}

void KvCache::advance(int n) {
  if (seq_len_ + n > max_seq_len_) {
    std::fprintf(stderr, "tinyqwen: KV cache overflow: seq_len %d + %d > max %d\n", seq_len_,
                 n, max_seq_len_);
    std::abort();
  }
  seq_len_ += n;
}

}  // namespace tinyqwen
