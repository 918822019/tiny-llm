// decode 阶段 attention：当前 token 的 query，对 KV cache 里所有历史位置做加权求和。
//
// 直观理解：当前 token 拿自己的 q，去和每个历史位置的 k 算"相关度"（点积），
// 用 softmax 把相关度变成一组加起来为 1 的权重，再用这组权重对历史 v 加权
// 求和——"把和我最相关的历史信息汇总过来"。这就是 attention。
//
// GQA（分组查询注意力）：q 有很多 head，但 k/v 的 head 更少，多个 q head
// 共享同一个 kv head，省显存。q head h 对应的 kv head 是 h / (n_heads/n_kv_heads)。
//
// 本实现用 online softmax 递推（和 FlashDecoding 同一思路）：
//   普通做法要先把所有位置的 score 存下来再 softmax（需要 O(seq_len) 暂存）；
//   online softmax 边扫边更新"当前最大值 m、指数和 l、加权和 oh"，只需 O(1)
//   暂存。这样 reference 零分配，访问模式也和后续融合 kernel 一致。

#include "ref_ops.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace tinyqwen {

void attention_decode_ref(const float* q, const float* k_cache, const float* v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float* out) {
  const int heads_per_kv = n_heads / n_kv_heads;  // 每个 kv head 被几个 q head 共享
  // 同一个 kv head 内部，相邻两个序列位置间隔 head_dim 个元素；
  // 一层 kv 块的布局是 [n_kv_heads][max_seq_len][head_dim]。
  const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

  // 逐个 q head 计算。
  for (int h = 0; h < n_heads; ++h) {
    const int kv = h / heads_per_kv;  // 当前 q head 用哪个 kv head
    const float* qh = q + static_cast<size_t>(h) * head_dim;
    const float* kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
    const float* vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
    float* oh = out + static_cast<size_t>(h) * head_dim;

    // online softmax 的三个状态量。处理完位置 [0, t) 后保持如下不变式：
    //   m  = 已见 score 的最大值（用来做数值稳定，防止 exp 上溢）
    //   l  = sum_j exp(s_j - m)（归一化分母）
    //   oh = sum_j exp(s_j - m) * v_j（未归一化的加权和，即输出）
    // 遇到更大的 score 时，旧的累加量要乘 exp(m - m_new) 重新标定。
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.0f;
    for (int i = 0; i < head_dim; ++i) oh[i] = 0.0f;

    // 扫描该 kv head 下所有有效位置 [0, seq_len)。
    for (int t = 0; t < seq_len; ++t) {
      const float* kt = kh + static_cast<size_t>(t) * head_dim;
      const float* vt = vh + static_cast<size_t>(t) * head_dim;

      // 相关度 = q 和 k 的点积，再乘缩放系数 scale。
      double dot = 0.0;
      for (int i = 0; i < head_dim; ++i) dot += static_cast<double>(qh[i]) * kt[i];
      const float s = static_cast<float>(dot) * scale;

      const float m_new = s > m ? s : m;          // 更新最大值
      const float rescale = std::exp(m - m_new);  // 旧累加量的重标定系数
      const float p = std::exp(s - m_new);        // 当前位置的未归一化权重
      for (int i = 0; i < head_dim; ++i) oh[i] = oh[i] * rescale + p * vt[i];
      l = l * rescale + p;
      m = m_new;
    }

    // 归一化：除以指数和 l，得到最终的加权平均输出。
    const float inv_l = 1.0f / l;  // decode 时恒有 seq_len >= 1，不会除零
    for (int i = 0; i < head_dim; ++i) oh[i] *= inv_l;
  }
}

}  // namespace tinyqwen
