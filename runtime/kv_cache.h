#pragma once

#include <cstddef>
#include <vector>

namespace tinyqwen {

// K/V cache for batch = 1, decode-style appending, fp32.
//
// Layout (K block first, then V block):
//   K: [n_layers][n_kv_heads][max_seq_len][head_dim]
//   V: [n_layers][n_kv_heads][max_seq_len][head_dim]
//
// v1 notes:
//   - positions [0, seq_len) are valid;
//   - speculative rollback is NOT implemented; future extension point is a
//     truncate_to(new_seq_len) on this class, no layout change needed.
class KvCache {
 public:
  KvCache() = default;
  KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

  void init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);
  void reset();  // seq_len = 0, keeps allocation

  int seq_len() const { return seq_len_; }
  int max_seq_len() const { return max_seq_len_; }
  int n_layers() const { return n_layers_; }
  int n_kv_heads() const { return n_kv_heads_; }
  int head_dim() const { return head_dim_; }

  // Whole-layer block pointers: [n_kv_heads][max_seq_len][head_dim].
  float* k(int layer);
  float* v(int layer);
  const float* k(int layer) const;
  const float* v(int layer) const;

  // seq_len += n; aborts if it would exceed max_seq_len (fail loud in v1).
  void advance(int n);

  size_t memory_bytes() const { return data_.size() * sizeof(float); }

 private:
  int n_layers_ = 0;
  int n_kv_heads_ = 0;
  int max_seq_len_ = 0;
  int head_dim_ = 0;
  int seq_len_ = 0;
  size_t layer_stride_ = 0;  // floats per one-layer block
  std::vector<float> data_;  // 2 * n_layers * layer_stride_ floats
};

}  // namespace tinyqwen
