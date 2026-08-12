// KV cache storage: one contiguous fp32 arena holding two blocks,
//   [ K: n_layers planes | V: n_layers planes ]
// where each plane is [n_kv_heads][max_seq_len][head_dim].
// Appends write slot `seq_len` of each plane; advance() commits the slot(s).
// The flat arena keeps a future mmap/quantized swap simple and avoids
// per-layer allocations.

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
  // 2x for the K and V blocks; zero-init so unwritten slots are visible
  // as zeros if an index bug ever reads them.
  data_.assign(2 * static_cast<size_t>(n_layers) * layer_stride_, 0.0f);
}

void KvCache::reset() { seq_len_ = 0; }

// K planes occupy slots [0, n_layers); V planes follow at
// [n_layers, 2*n_layers) inside the same arena.
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
