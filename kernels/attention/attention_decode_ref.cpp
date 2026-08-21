// ============================================================================
// attention_decode_ref.cpp — Decode 阶段 GQA Attention 的参考实现（标量版）
// ============================================================================
// 本文件实现 decode 阶段的 Grouped Query Attention (GQA) 算子。
//
// 数学定义：
//   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
//   其中 Q 是当前 token 的 query 向量 [n_heads, head_dim]，
//   K/V 来自 KV Cache，形状为 [n_kv_heads, seq_len, head_dim]。
//   scale = 1/sqrt(head_dim)，防止点积过大导致 softmax 梯度消失。
//
// 在 Transformer 中的位置：
//   每个 decoder layer 的核心计算，位于 input_layernorm 之后、FFN 之前。
//   decode 阶段每次只处理一个 token（seq_len_q = 1），所以 q 是一行向量，
//   需要对 KV cache 中所有历史位置做加权求和。
//
// ref 与 neon 版本的关系：
//   - 本文件是标量参考实现，作为正确性锚点（golden reference）。
//   - attention_decode_neon.cpp 将内循环向量化（NEON intrinsic），
//     但算法结构、online softmax 状态更新逻辑完全一致。
//   - NEON 版用 float 累加点积（ref 用 double），数值差异 ~1e-7。
//
// 优化要点（本参考实现）：
//   1. Online Softmax：不需要 O(seq_len) 暂存 score 数组，只用三个状态量
//      (m, l, oh) 边扫描边更新，零额外内存分配。这和 FlashDecoding 同一思路。
//   2. GQA 映射：多个 q head 共享同一个 kv head，通过 heads_per_kv 分组。
//   3. KV Cache 布局：[n_kv_heads][max_seq_len][head_dim]，连续存储利于缓存。
// ============================================================================

#include "ref_ops.h" // 声明 attention_decode_ref 等参考算子的头文件

#include <cmath>    // std::exp, std::numeric_limits
#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits<float>::infinity()

namespace tinyqwen {
  // =========================================================================
  // attention_decode_ref — Decode 阶段 GQA Attention 参考实现
  // =========================================================================
  // 功能：给定当前 token 的 query q，对 KV cache 中所有历史位置执行
  //       grouped query attention，输出加权求和结果。
  // 参数：
  //   q          — 当前 token 的 query 向量，形状 [n_heads, head_dim]，行主序
  //   k_cache    — KV cache 中 key 的起始地址，布局 [n_kv_heads][max_seq_len][head_dim]
  //   v_cache    — KV cache 中 value 的起始地址，布局同上
  //   seq_len    — 当前有效序列长度（已缓存的历史 token 数）
  //   max_seq_len— KV cache 的最大容量（用于计算层间 stride）
  //   n_heads    — query 的注意力头数量
  //   n_kv_heads — key/value 的注意力头数量（GQA 中 <= n_heads）
  //   head_dim   — 每个注意力头的维度（Qwen2.5-0.5B 为 64）
  //   scale      — 缩放因子，通常为 1/sqrt(head_dim)
  //   out        — 输出缓冲区，形状 [n_heads, head_dim]
  void attention_decode_ref(const float *q, const float *k_cache, const float *v_cache,
                            int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                            int head_dim, float scale, float *out) {
    // 计算每个 kv head 被几个 q head 共享（GQA 的分组比）
    // 例如 n_heads=14, n_kv_heads=2 → heads_per_kv=7，即每 7 个 q head 共享 1 个 kv head
    const int heads_per_kv = n_heads / n_kv_heads;
    // 同一个 kv head 内部，相邻两个序列位置间隔 head_dim 个元素；
    // 一层 kv 块的布局是 [n_kv_heads][max_seq_len][head_dim]，
    // 因此跨过一个 kv head 需要跳过 max_seq_len * head_dim 个 float
    const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

    // 逐个 q head 计算 attention
    for (int h = 0; h < n_heads; ++h) {
      // GQA 映射：当前 q head h 对应的 kv head 编号 = h / heads_per_kv
      const int kv = h / heads_per_kv;
      // 当前 q head 的 query 向量起始地址：q[h * head_dim .. (h+1) * head_dim)
      const float *qh = q + static_cast<size_t>(h) * head_dim;
      // 对应 kv head 的 key 缓存起始地址：k_cache[kv * kv_layer_stride ..]
      const float *kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
      // 对应 kv head 的 value 缓存起始地址：v_cache[kv * kv_layer_stride ..]
      const float *vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
      // 当前 q head 的输出写入位置：out[h * head_dim .. (h+1) * head_dim)
      float *oh = out + static_cast<size_t>(h) * head_dim;

      // ---- Online Softmax 的三个状态量 ----
      // 处理完位置 [0, t) 后保持如下不变式：
      //   m  = 已见 score 的最大值（用来做数值稳定，防止 exp 上溢）
      //   l  = sum_j exp(s_j - m)（归一化分母，即 softmax 的分母）
      //   oh = sum_j exp(s_j - m) * v_j（未归一化的加权和，即输出的分子部分）
      // 初始时没有见过任何位置，m 设为负无穷（确保第一个 score 一定成为新最大值）
      float m = -std::numeric_limits<float>::infinity();
      // 分母初始为 0（还没累加任何权重）
      float l = 0.0f;
      // 输出累加器初始化为全零（还没有任何位置的贡献）
      for (int i = 0; i < head_dim; ++i) {
        oh[i] = 0.0f;
      }

      // 扫描该 kv head 下所有有效位置 [0, seq_len)
      for (int t = 0; t < seq_len; ++t) {
        // 当前位置 t 的 key 向量：kh[t * head_dim .. (t+1) * head_dim)
        const float *kt = kh + static_cast<size_t>(t) * head_dim;
        // 当前位置 t 的 value 向量：vh[t * head_dim .. (t+1) * head_dim)
        const float *vt = vh + static_cast<size_t>(t) * head_dim;

        // 计算相关度（attention score）= q 和 k 的点积，再乘缩放系数 scale
        // 使用 double 累加减少浮点误差（head_dim=64 次乘法累加，float 精度不够）
        double dot = 0.0;
        for (int i = 0; i < head_dim; ++i) {
          // 将 q[i] 提升到 double 精度后再与 k[i] 相乘并累加
          const double qi = static_cast<double>(qh[i]);
          dot += qi * kt[i];
        }
        // 将 double 点积转回 float，乘以缩放系数得到最终 score
        const float s = static_cast<float>(dot) * scale;

        // ---- Online Softmax 状态更新 ----
        // 新的最大值 = max(旧最大值, 当前 score)
        const float m_new = s > m ? s : m;
        // 重标定系数：当最大值从 m 变为 m_new 时，旧的累加量需要乘以 exp(m - m_new)
        // 这保证了 exp(score - m_new) 的一致性——所有历史项和当前项都相对于同一个基准
        const float rescale = std::exp(m - m_new);
        // 当前位置的未归一化权重 = exp(s - m_new)，始终 <= 1（因为 m_new >= s）
        const float p = std::exp(s - m_new);

        // 更新输出累加器：oh = oh * rescale + p * vt（逐元素）
        // rescale 把旧的 oh 调整到新基准 m_new 下，然后加上当前位置的贡献
        for (int i = 0; i < head_dim; ++i) {
          // 旧累加值按新旧 max 之差重标定
          const float rescaled_old = oh[i] * rescale;
          // 当前位置对该维度的贡献 = 未归一化权重 * value
          const float contrib = p * vt[i];
          // 合并：重标定后的旧值 + 新贡献
          oh[i] = rescaled_old + contrib;
        }

        // 更新分母 l：同样先重标定再累加当前权重
        const float rescaled_l = l * rescale; // 旧分母按新基准重标定
        l = rescaled_l + p;                   // 加上当前未归一化权重
        m = m_new;                            // 更新最大值为新值
      }

      // 归一化：除以指数和 l，得到最终的加权平均输出
      // decode 时恒有 seq_len >= 1，所以 l > 0，不会除零
      const float inv_l = 1.0f / l;
      // 每个维度除以 l，完成 softmax 归一化
      for (int i = 0; i < head_dim; ++i) {
        oh[i] = oh[i] * inv_l;
      }
    }
  }
} // namespace tinyqwen
