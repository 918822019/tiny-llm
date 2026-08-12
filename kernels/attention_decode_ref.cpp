// decode 阶段 attention：单个 query 位置对已缓存的 KV 做注意力。
//
// 采用 online softmax 递推（与 FlashDecoding 同一思路），kernel 只需
// O(1) 暂存内存，不需要逐位置的 score 缓冲。这样 reference 实现零分配，
// 访问模式也已经与后续融合 kernel 一致。

#include "ref_ops.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace tinyqwen {

void attention_decode_ref(const float* q, const float* k_cache, const float* v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float* out) {
  const int heads_per_kv = n_heads / n_kv_heads;  // GQA 扇出系数
  // 同一层块内相邻两个 kv head 之间的步长：
  // 布局为 [n_kv_heads][max_seq_len][head_dim]。
  const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

  for (int h = 0; h < n_heads; ++h) {
    const int kv = h / heads_per_kv;
    const float* qh = q + static_cast<size_t>(h) * head_dim;
    const float* kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
    const float* vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
    float* oh = out + static_cast<size_t>(h) * head_dim;

    // online softmax 状态。处理完位置 [0, t) 后的不变式：
    //   m  = 已见 score 的最大值
    //   l  = sum_j exp(s_j - m)
    //   oh = sum_j exp(s_j - m) * v_j      （未归一化的输出）
    // 出现新的最大值 m_new 时，旧累加量乘以 exp(m - m_new) 重标定；
    // 在第一个真实 score 到来之前（m == -inf），该系数恰为 1。
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.0f;
    for (int i = 0; i < head_dim; ++i) oh[i] = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
      const float* kt = kh + static_cast<size_t>(t) * head_dim;
      const float* vt = vh + static_cast<size_t>(t) * head_dim;

      double dot = 0.0;
      for (int i = 0; i < head_dim; ++i) dot += static_cast<double>(qh[i]) * kt[i];
      const float s = static_cast<float>(dot) * scale;

      const float m_new = s > m ? s : m;
      const float rescale = std::exp(m - m_new);  // m == -inf 时为 1.0f
      const float p = std::exp(s - m_new);
      for (int i = 0; i < head_dim; ++i) oh[i] = oh[i] * rescale + p * vt[i];
      l = l * rescale + p;
      m = m_new;
    }

    const float inv_l = 1.0f / l;  // decode 期间恒有 seq_len >= 1
    for (int i = 0; i < head_dim; ++i) oh[i] *= inv_l;
  }
}

}  // namespace tinyqwen
