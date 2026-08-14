// decode attention 的 NEON 版（aarch64）。
//
// 结构与 attention_decode_ref 完全一致：逐 q head、online softmax 三状态量
// (m, l, oh) 边扫边更新、零暂存、GQA 映射、KV 布局 [n_kv_heads][max_seq_len]
// [head_dim] 都不变。NEON 化的只有两处内循环：
//   1. q·k_t 点积：ref 用 double 累加，这里用 4 路 float32x4 累加器（ulp 级差异）。
//   2. oh = oh*rescale + p*v_t：vmulq 缩放 + vfmaq 累加，一条指令 4 lane。
//   3. 归一化 oh *= 1/l 也向量化。
//
// exp / online softmax 的标量状态更新保持标量（每步每 head 只算一次，不是热点）。
// 数值差异仅来自点积的 float vs double 累加（~1e-7 相对），单测按容差门禁
// （rel 1e-4）对齐 ref；贪心决策不受影响（golden token 逐位一致）。

#include "dispatch.h" // TINYQWEN_ATTENTION_DECODE_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace tinyqwen {
  namespace {
    // NEON 点积：dot = sum_i q[i]*k[i]（4 路累加器 + 标量尾段）。
    inline float dot_neon(const float *q, const float *k, int n) {
      float32x4_t a0 = vdupq_n_f32(0.0f);
      float32x4_t a1 = vdupq_n_f32(0.0f);
      float32x4_t a2 = vdupq_n_f32(0.0f);
      float32x4_t a3 = vdupq_n_f32(0.0f);
      int i = 0;
      for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(q + i), vld1q_f32(k + i));
        a1 = vfmaq_f32(a1, vld1q_f32(q + i + 4), vld1q_f32(k + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(q + i + 8), vld1q_f32(k + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(q + i + 12), vld1q_f32(k + i + 12));
      }
      for (; i + 4 <= n; i += 4) {
        a0 = vfmaq_f32(a0, vld1q_f32(q + i), vld1q_f32(k + i));
      }
      float dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
      for (; i < n; ++i) {
        dot += q[i] * k[i];
      }
      return dot;
    }

    void attention_decode_neon(const float *q, const float *k_cache, const float *v_cache,
                               int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                               int head_dim, float scale, float *out) {
      const int heads_per_kv = n_heads / n_kv_heads;
      const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

      for (int h = 0; h < n_heads; ++h) {
        const int kv = h / heads_per_kv;
        const float *qh = q + static_cast<size_t>(h) * head_dim;
        const float *kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
        const float *vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
        float *oh = out + static_cast<size_t>(h) * head_dim;

        float m = -std::numeric_limits<float>::infinity();
        float l = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
          oh[i] = 0.0f;
        }

        for (int t = 0; t < seq_len; ++t) {
          const float *kt = kh + static_cast<size_t>(t) * head_dim;
          const float *vt = vh + static_cast<size_t>(t) * head_dim;

          const float s = dot_neon(qh, kt, head_dim) * scale;

          const float m_new = s > m ? s : m;
          const float rescale = std::exp(m - m_new);
          const float p = std::exp(s - m_new);

          // oh = oh*rescale + p*vt（向量化）。
          const float32x4_t vr = vdupq_n_f32(rescale);
          const float32x4_t vp = vdupq_n_f32(p);
          int i = 0;
          for (; i + 4 <= head_dim; i += 4) {
            float32x4_t o = vld1q_f32(oh + i);
            o = vmulq_f32(o, vr);                 // oh * rescale
            o = vfmaq_f32(o, vp, vld1q_f32(vt + i)); // + p * vt
            vst1q_f32(oh + i, o);
          }
          for (; i < head_dim; ++i) {
            oh[i] = oh[i] * rescale + p * vt[i];
          }

          l = l * rescale + p;
          m = m_new;
        }

        // 归一化 oh *= 1/l（向量化）。
        const float inv_l = 1.0f / l;
        const float32x4_t vi = vdupq_n_f32(inv_l);
        int i = 0;
        for (; i + 4 <= head_dim; i += 4) {
          vst1q_f32(oh + i, vmulq_f32(vld1q_f32(oh + i), vi));
        }
        for (; i < head_dim; ++i) {
          oh[i] *= inv_l;
        }
      }
    }
  } // namespace

  // 自注册进 attention 注册表，名 "neon"。仅 aarch64 构建存在。
  TINYQWEN_ATTENTION_DECODE_VARIANT(attention_decode_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
