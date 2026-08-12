#pragma once

// KV cache：decode 阶段的记忆。先读 docs/infra_primer.md 第 3、6 节。
//
// 为什么需要它？attention 里每个新 token 都要看之前所有 token。如果每生成
// 一个词都重算前面所有词，计算量平方级爆炸。所以把历史 token 算出的
// K/V 向量缓存起来复用——这就是 KV cache。

#include <cstddef>
#include <vector>

namespace tinyqwen {

// batch = 1、decode 式追加的 K/V 缓存，fp32。
//
// 内存布局（先一整块 K，接着一整块 V）：
//   K: [n_layers][n_kv_heads][max_seq_len][head_dim]
//   V: [n_layers][n_kv_heads][max_seq_len][head_dim]
//
// 读法：第 1 维是层号，第 2 维是 kv 头号，第 3 维是序列位置，第 4 维是头内维度。
// 访问某层 l、某头 h、某位置 t 的起点偏移（元素个数）是：
//   l * (n_kv_heads * max_seq_len * head_dim) + h * (max_seq_len * head_dim) + t * head_dim
//
// v1 说明：
//   - 已填充的位置是 [0, seq_len)；
//   - speculative rollback 未实现；将来的扩展点是加一个 truncate_to(new_len)，
//     布局不用改。
class KvCache {
 public:
  KvCache() = default;
  KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

  // 一次性分配全部内存（见 .cpp 里的 arena 说明）。
  void init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);
  void reset();  // 逻辑上清空（seq_len 归零），但保留已分配的内存

  int seq_len() const { return seq_len_; }        // 当前已缓存多少个位置
  int max_seq_len() const { return max_seq_len_; }  // 容量上限
  int n_layers() const { return n_layers_; }
  int n_kv_heads() const { return n_kv_heads_; }
  int head_dim() const { return head_dim_; }

  // 取某一层的整块 K / V 指针：[n_kv_heads][max_seq_len][head_dim]。
  // 调用方拿到指针后自己按上面的偏移公式找具体位置。
  float* k(int layer);
  float* v(int layer);
  const float* k(int layer) const;
  const float* v(int layer) const;

  // seq_len += n。超过容量直接 abort（fail fast，见 primer 第 9 节）。
  void advance(int n);

  // 这块缓存总共占多少字节（打印内存占用用）。
  size_t memory_bytes() const { return data_.size() * sizeof(float); }

 private:
  int n_layers_ = 0;
  int n_kv_heads_ = 0;
  int max_seq_len_ = 0;
  int head_dim_ = 0;
  int seq_len_ = 0;
  size_t layer_stride_ = 0;  // 一层占多少个 float（= n_kv_heads*max_seq_len*head_dim）
  std::vector<float> data_;  // 一整块 arena：2 * n_layers * layer_stride_ 个 float
};

}  // namespace tinyqwen
