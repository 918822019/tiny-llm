#pragma once

#include <cstddef>
#include <vector>

namespace tinyqwen {

// batch = 1、decode 式追加的 K/V 缓存，fp32。
//
// 布局（先是 K 块，然后是 V 块）：
//   K: [n_layers][n_kv_heads][max_seq_len][head_dim]
//   V: [n_layers][n_kv_heads][max_seq_len][head_dim]
//
// v1 说明：
//   - 有效位置为 [0, seq_len)；
//   - speculative rollback 未实现；将来的扩展点是在本类上加
//     truncate_to(new_seq_len)，布局无需改动。
class KvCache {
 public:
  KvCache() = default;
  KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

  void init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);
  void reset();  // seq_len = 0，保留已分配的内存

  int seq_len() const { return seq_len_; }
  int max_seq_len() const { return max_seq_len_; }
  int n_layers() const { return n_layers_; }
  int n_kv_heads() const { return n_kv_heads_; }
  int head_dim() const { return head_dim_; }

  // 整层块指针：[n_kv_heads][max_seq_len][head_dim]。
  float* k(int layer);
  float* v(int layer);
  const float* k(int layer) const;
  const float* v(int layer) const;

  // seq_len += n；超过 max_seq_len 直接 abort（v1 大声失败）。
  void advance(int n);

  size_t memory_bytes() const { return data_.size() * sizeof(float); }

 private:
  int n_layers_ = 0;
  int n_kv_heads_ = 0;
  int max_seq_len_ = 0;
  int head_dim_ = 0;
  int seq_len_ = 0;
  size_t layer_stride_ = 0;  // 单层块的 float 数
  std::vector<float> data_;  // 2 * n_layers * layer_stride_ 个 float
};

}  // namespace tinyqwen
