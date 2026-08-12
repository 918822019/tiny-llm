// KV cache 存储：一块连续的 fp32 arena，内含两个块：
//   [ K: n_layers 个平面 | V: n_layers 个平面 ]
// 每个平面为 [n_kv_heads][max_seq_len][head_dim]。
// 追加时写每个平面的第 seq_len 个槽位；advance() 提交槽位。
// 单块 arena 让将来的 mmap / 量化换入保持简单，也避免逐层分配。

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
  layer_stride_ = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
  // 乘 2 是 K、V 两个块；清零初始化，一旦索引写错读到未写槽位，
  // 看到的是 0 而不是脏数据。
  data_.assign(2 * static_cast<size_t>(n_layers) * layer_stride_, 0.0f);
}

void KvCache::reset() { seq_len_ = 0; }

// K 平面占 arena 的 [0, n_layers)，V 平面紧随其后位于
// [n_layers, 2*n_layers)。
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
