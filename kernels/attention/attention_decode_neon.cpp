// ============================================================================
// attention_decode_neon.cpp — Decode 阶段 GQA Attention 的 NEON SIMD 优化版
// ============================================================================
// 本文件实现 decode 阶段 Grouped Query Attention 的 ARM NEON 向量化版本。
//
// 数学定义与 attention_decode_ref.cpp 完全相同：
//   Attention(Q, K, V) = softmax(Q·K^T / sqrt(d)) · V
//   Online Softmax 三状态量 (m, l, oh) 边扫描边更新，零暂存。
//
// 在 Transformer 中的位置：
//   与 ref 版相同——每个 decoder layer 的核心计算，decode 阶段逐 token 调用。
//
// ref 与 neon 版本的关系：
//   - 算法结构、GQA 映射、KV Cache 布局、online softmax 逻辑完全一致。
//   - NEON 化的只有两处内循环：
//     1. q·k_t 点积：ref 用 double 累加，这里用 4 路 float32x4 累加器（ulp 级差异）。
//     2. oh = oh*rescale + p*v_t：vmulq 缩放 + vfmaq 累加，一条指令处理 4 lane。
//     3. 归一化 oh *= 1/l 也向量化。
//   - exp / online softmax 的标量状态更新保持标量（每步每 head 只算一次，不是热点）。
//
// 优化要点：
//   1. dot_neon 使用 4 路独立累加器 + 循环展开到 16 元素/轮，隐藏 FMA 延迟。
//   2. oh 更新用 vmulq + vfmaq 组合，避免标量循环。
//   3. 归一化用 vdupq_n_f32 广播 inv_l 后 vmulq 批量乘。
//   4. 数值差异仅来自点积的 float vs double 累加（~1e-7 相对），
//      单测按容差门禁（rel 1e-4）对齐 ref；贪心决策不受影响。
// ============================================================================

#include "dispatch.h" // TINYQWEN_ATTENTION_DECODE_VARIANT 自注册宏，用于将本变体注册到 dispatch 表
#include "ref_ops.h"  // 声明 attention_decode 函数签名

#if defined(__aarch64__) || defined(_M_ARM64) // 仅在 ARM64 平台编译以下代码

#include <arm_neon.h> // ARM NEON SIMD intrinsic 头文件

#include <cmath>    // std::exp
#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits<float>::infinity()

namespace tinyqwen {
  namespace { // 匿名命名空间：内部辅助函数不暴露到外部链接
    // =========================================================================
    // dot_neon — NEON 向量化的点积计算
    // =========================================================================
    // 功能：计算两个长度为 n 的 float 数组的点积 sum_i(q[i] * k[i])
    // 参数：
    //   q — 第一个向量（query）
    //   k — 第二个向量（key）
    //   n — 向量长度（通常为 head_dim = 64）
    // 返回值：点积结果（float 精度）
    // 说明：使用 4 路独立的 float32x4 累加器，主循环每轮处理 16 个元素，
    //       隐藏 vfmaq 的流水线延迟。不足 16 的部分按 4 个一组补齐，
    //       最后不足 4 的用标量处理。
    inline float dot_neon(const float *q, const float *k, int n) {
      // 4 路独立的 128-bit 累加器，初始化为全零
      float32x4_t a0 = vdupq_n_f32(0.0f); // 累加器 0：处理第 0, 16, 32... 组 4 元素
      float32x4_t a1 = vdupq_n_f32(0.0f); // 累加器 1：处理第 4, 20, 36... 组 4 元素
      float32x4_t a2 = vdupq_n_f32(0.0f); // 累加器 2：处理第 8, 24, 40... 组 4 元素
      float32x4_t a3 = vdupq_n_f32(0.0f); // 累加器 3：处理第 12, 28, 44... 组 4 元素
      int i = 0;
      // 主循环：每轮处理 16 个元素（4 组 × 4 lane），充分利用 NEON 流水线
      for (; i + 16 <= n; i += 16) {
        // vld1q_f32: 从内存加载 4 个连续 float 到 128-bit NEON 寄存器
        // vfmaq_f32(a, b, c): fused multiply-add，a += b * c（4 lane 并行）
        a0 = vfmaq_f32(a0, vld1q_f32(q + i), vld1q_f32(k + i));         // 第 0-3 个元素
        a1 = vfmaq_f32(a1, vld1q_f32(q + i + 4), vld1q_f32(k + i + 4)); // 第 4-7 个元素
        a2 = vfmaq_f32(a2, vld1q_f32(q + i + 8), vld1q_f32(k + i + 8)); // 第 8-11 个元素
        a3 = vfmaq_f32(a3, vld1q_f32(q + i + 12), vld1q_f32(k + i + 12)); // 第 12-15 个元素
      }
      // 次循环：处理剩余不足 16 但 >= 4 的元素（每次 4 个）
      for (; i + 4 <= n; i += 4) {
        a0 = vfmaq_f32(a0, vld1q_f32(q + i), vld1q_f32(k + i));
      }
      // 横向归约：将 4 路累加器合并为一个标量
      // vaddq_f32: 两两相加（4 lane → 4 lane）
      // vaddvq_f32: 将 4 个 lane 求和为单个 float
      float dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
      // 标量尾段：处理最后不足 4 个的元素
      for (; i < n; ++i) {
        dot += q[i] * k[i];
      }
      return dot;
    }

    // =========================================================================
    // attention_decode_neon — Decode 阶段 GQA Attention 的 NEON 实现
    // =========================================================================
    // 功能：与 attention_decode_ref 完全相同，仅内循环向量化
    // 参数：同 attention_decode_ref
    void attention_decode_neon(const float *q, const float *k_cache, const float *v_cache,
                               int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                               int head_dim, float scale, float *out) {
      // GQA 分组比：每个 kv head 被几个 q head 共享
      const int heads_per_kv = n_heads / n_kv_heads;
      // KV cache 中跨过一个 kv head 需要的 stride = max_seq_len * head_dim
      const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

      // 逐个 q head 计算 attention（与 ref 结构一致）
      for (int h = 0; h < n_heads; ++h) {
        // GQA 映射：当前 q head h 对应的 kv head 编号
        const int kv = h / heads_per_kv;
        // 当前 q head 的 query 向量起始地址
        const float *qh = q + static_cast<size_t>(h) * head_dim;
        // 对应 kv head 的 key 缓存起始地址
        const float *kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
        // 对应 kv head 的 value 缓存起始地址
        const float *vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
        // 输出写入位置
        float *oh = out + static_cast<size_t>(h) * head_dim;

        // Online Softmax 三状态量初始化（与 ref 完全一致）
        float m = -std::numeric_limits<float>::infinity(); // 最大值初始为负无穷
        float l = 0.0f;                                     // 分母初始为零
        // 输出累加器初始化为全零
        for (int i = 0; i < head_dim; ++i) {
          oh[i] = 0.0f;
        }

        // 扫描所有有效位置 [0, seq_len)
        for (int t = 0; t < seq_len; ++t) {
          // 当前位置 t 的 key/value 向量地址
          const float *kt = kh + static_cast<size_t>(t) * head_dim;
          const float *vt = vh + static_cast<size_t>(t) * head_dim;

          // 用 NEON 点积代替 ref 的 double 累加点积，再乘缩放系数
          const float s = dot_neon(qh, kt, head_dim) * scale;

          // Online Softmax 状态更新（标量，每步只算一次，不是热点）
          const float m_new = s > m ? s : m;           // 更新最大值
          const float rescale = std::exp(m - m_new);    // 旧累加量的重标定系数
          const float p = std::exp(s - m_new);          // 当前位置的未归一化权重

          // ---- oh = oh*rescale + p*vt（NEON 向量化）----
          // 将标量 rescale 和 p 广播到 128-bit NEON 寄存器的 4 个 lane
          const float32x4_t vr = vdupq_n_f32(rescale); // rescale 广播到 4 lane
          const float32x4_t vp = vdupq_n_f32(p);       // p 广播到 4 lane
          int i = 0;
          // 向量化循环：每次处理 4 个维度
          for (; i + 4 <= head_dim; i += 4) {
            // 从内存加载当前输出的 4 个分量
            float32x4_t o = vld1q_f32(oh + i);
            // vmulq_f32: 4 lane 并行乘法，o = oh * rescale（重标定旧值）
            o = vmulq_f32(o, vr);
            // vfmaq_f32: fused multiply-add，o += p * vt（累加新贡献）
            o = vfmaq_f32(o, vp, vld1q_f32(vt + i));
            // vst1q_f32: 将 4 个结果写回内存
            vst1q_f32(oh + i, o);
          }
          // 标量尾段：处理不足 4 个的剩余维度
          for (; i < head_dim; ++i) {
            oh[i] = oh[i] * rescale + p * vt[i];
          }

          // 更新分母 l（标量）
          l = l * rescale + p;
          // 更新最大值
          m = m_new;
        }

        // ---- 归一化 oh *= 1/l（NEON 向量化）----
        // decode 时恒有 seq_len >= 1，l > 0，不会除零
        const float inv_l = 1.0f / l;
        // 将 inv_l 广播到 4 lane
        const float32x4_t vi = vdupq_n_f32(inv_l);
        int i = 0;
        // 向量化归一化：每次处理 4 个维度
        for (; i + 4 <= head_dim; i += 4) {
          // vld1q_f32 加载 → vmulq_f32 乘以 inv_l → vst1q_f32 写回
          vst1q_f32(oh + i, vmulq_f32(vld1q_f32(oh + i), vi));
        }
        // 标量尾段：处理剩余维度
        for (; i < head_dim; ++i) {
          oh[i] *= inv_l;
        }
      }
    }
  } // namespace

  // =========================================================================
  // dot_f16_neon — query(fp32) 与 key(fp16) 的点积，寄存器内转 fp32
  // =========================================================================
  // 功能：sum_i(q[i] * k[i])，其中 q 是 fp32，k 是 fp16（uint16_t*）。
  // 关键：k 从内存按 fp16 加载（vld1_f16，4 个/次），用 vcvt_f32_f16 在寄存器
  // 内转 fp32，再做 FMA。没有独立的反量化遍历。
  inline float dot_f16_neon(const float *q, const uint16_t *k, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int i = 0;
    // 主循环：每轮 16 元素（4 组 × 4 lane）
    for (; i + 16 <= n; i += 16) {
      a0 = vfmaq_f32(a0, vld1q_f32(q + i),
                     vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(k + i))));
      a1 = vfmaq_f32(a1, vld1q_f32(q + i + 4),
                     vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(k + i + 4))));
      a2 = vfmaq_f32(a2, vld1q_f32(q + i + 8),
                     vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(k + i + 8))));
      a3 = vfmaq_f32(a3, vld1q_f32(q + i + 12),
                     vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(k + i + 12))));
    }
    for (; i + 4 <= n; i += 4) {
      a0 = vfmaq_f32(a0, vld1q_f32(q + i),
                     vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(k + i))));
    }
    float dot = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; ++i) {
      dot += q[i] * half_to_float(k[i]);
    }
    return dot;
  }

  // =========================================================================
  // attention_decode_f16kv_neon — fp16-KV 融合 attention（GQA，online softmax）
  // =========================================================================
  // 与 attention_decode_neon 完全相同的数学，仅 K/V 从 fp16 加载、寄存器内转
  // fp32。消灭独立反量化遍历。
  void attention_decode_f16kv_neon(const float *q, const uint16_t *k_cache,
                                   const uint16_t *v_cache, int seq_len, int max_seq_len,
                                   int n_heads, int n_kv_heads, int head_dim, float scale,
                                   float *out) {
    const int heads_per_kv = n_heads / n_kv_heads;
    const size_t kv_layer_stride = static_cast<size_t>(max_seq_len) * head_dim;

    for (int h = 0; h < n_heads; ++h) {
      const int kv = h / heads_per_kv;
      const float *qh = q + static_cast<size_t>(h) * head_dim;
      const uint16_t *kh = k_cache + static_cast<size_t>(kv) * kv_layer_stride;
      const uint16_t *vh = v_cache + static_cast<size_t>(kv) * kv_layer_stride;
      float *oh = out + static_cast<size_t>(h) * head_dim;

      float m = -std::numeric_limits<float>::infinity();
      float l = 0.0f;
      for (int i = 0; i < head_dim; ++i) oh[i] = 0.0f;

      for (int t = 0; t < seq_len; ++t) {
        const uint16_t *kt = kh + static_cast<size_t>(t) * head_dim;
        const uint16_t *vt = vh + static_cast<size_t>(t) * head_dim;

        const float s = dot_f16_neon(qh, kt, head_dim) * scale;
        const float m_new = s > m ? s : m;
        const float rescale = std::exp(m - m_new);
        const float p = std::exp(s - m_new);

        const float32x4_t vr = vdupq_n_f32(rescale);
        const float32x4_t vp = vdupq_n_f32(p);
        int i = 0;
        for (; i + 4 <= head_dim; i += 4) {
          float32x4_t o = vld1q_f32(oh + i);
          o = vmulq_f32(o, vr);
          // v 从 fp16 加载、寄存器内转 fp32，再累加
          float32x4_t vv = vcvt_f32_f16(vld1_f16(reinterpret_cast<const float16_t *>(vt + i)));
          o = vfmaq_f32(o, vp, vv);
          vst1q_f32(oh + i, o);
        }
        for (; i < head_dim; ++i) {
          oh[i] = oh[i] * rescale + p * half_to_float(vt[i]);
        }

        l = l * rescale + p;
        m = m_new;
      }

      const float inv_l = 1.0f / l;
      const float32x4_t vi = vdupq_n_f32(inv_l);
      int i = 0;
      for (; i + 4 <= head_dim; i += 4) {
        vst1q_f32(oh + i, vmulq_f32(vld1q_f32(oh + i), vi));
      }
      for (; i < head_dim; ++i) oh[i] *= inv_l;
    }
  }

  // 自注册进 attention 注册表，名称为 "neon"。仅在 aarch64 构建中存在。
  // TINYQWEN_ATTENTION_DECODE_VARIANT 宏会在 dispatch 表中注册此函数指针，
  // 运行时可通过 backend 选择机制自动选用。
  TINYQWEN_ATTENTION_DECODE_VARIANT(attention_decode_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
