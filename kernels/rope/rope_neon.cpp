// ============================================================================
// rope_neon.cpp — RoPE 旋转位置编码的 NEON SIMD 优化版（aarch64）
// ============================================================================
// 本文件实现 Rotary Position Embedding (RoPE) 的 ARM NEON 向量化版本。
//
// 数学定义与 rope_ref.cpp 完全相同：
//   x'[i]       = x[i] * cos(θ_i) - x[i + d/2] * sin(θ_i)
//   x'[i + d/2] = x[i + d/2] * cos(θ_i) + x[i] * sin(θ_i)
//   θ_i = pos * base^(-2i/d)
//
// 在 Transformer 中的位置：
//   与 ref 版相同——每个 decoder layer 中 Q/K 投影之后、attention 之前。
//
// ref 与 neon 版本的关系：
//   - cos/sin 表的计算与 ref 逐字相同（标量 libm），保证喂给旋转的数据一致。
//   - 仅"施加旋转"的内循环被 NEON 化：vmulq + vsubq/vaddq，一次处理 4 个配对。
//   - 乘加分离（不用 vfmaq），尽量贴近 ref 的逐元素舍入行为。
//
// 优化要点：
//   1. cos/sin 表仍用标量计算（half 次三角函数调用，不是热点）。
//   2. 旋转内循环用 NEON 一次处理 4 个配对（8 个分量），吞吐量提升 4 倍。
//   3. 使用 vmulq + vsubq/vaddq 而非 vfmaq，避免 FMA 引入额外的舍入差异。
//   4. 数值差异仅来自编译器可能对 ref 做 FP 收缩（fma）与本实现的乘加分离
//      之间的 ulp 级差别，单测按容差门禁（rel 1e-5）对齐 ref。
//   5. rotate-half 约定、GQA 的 head 遍历、pos/theta 语义与 ref 完全一致。
// ============================================================================

#include "dispatch.h" // TINYQWEN_ROPE_VARIANT 自注册宏
#include "ref_ops.h"  // 声明 rope 函数签名

#if defined(__aarch64__) || defined(_M_ARM64) // 仅在 ARM64 平台编译

#include <arm_neon.h> // ARM NEON SIMD intrinsic 头文件

#include <cmath>   // std::pow, std::cos, std::sin
#include <cstdio>  // std::fprintf
#include <cstdlib> // std::abort

namespace tinyqwen {
  namespace { // 匿名命名空间：内部函数不暴露到外部链接
    // =========================================================================
    // rope_neon — RoPE 旋转位置编码的 NEON 实现
    // =========================================================================
    // 功能：对 q 和 k 的所有 head 就地施加旋转位置编码（NEON 加速旋转部分）
    // 参数：同 rope_ref
    void rope_neon(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
                   float theta) {
      // half = head_dim / 2：配对的"前半"长度（rotate_half 约定）
      const int half = head_dim / 2;
      // 最大支持的 half 值，与 rope_ref 一致
      constexpr int kMaxHalf = 256;
      // 合法性检查
      if (half <= 0 || half > kMaxHalf) {
        std::fprintf(stderr, "rope_neon: unsupported head_dim %d\n", head_dim);
        std::abort(); // 不支持的 head_dim，终止程序
      }

      // ---- 预计算 cos/sin 表（与 ref 逐字相同，标量 libm）----
      // 所有 head 共享同一组 cos/sin 值
      float cs[kMaxHalf]; // cos 值数组
      float sn[kMaxHalf]; // sin 值数组
      for (int i = 0; i < half; ++i) {
        // inv_freq[i] = theta^(-2i/head_dim)：不同配对使用不同的频率
        const float exponent = -static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float inv_freq = std::pow(theta, exponent);     // 计算逆频率
        const float angle = static_cast<float>(pos) * inv_freq; // 旋转角度
        cs[i] = std::cos(angle); // 缓存 cos 值
        sn[i] = std::sin(angle); // 缓存 sin 值
      }

      // ---- 定义旋转 lambda：对单个 head 施加旋转（NEON 向量化）----
      const auto apply = [&](float *x) {
        int i = 0;
        // NEON 向量化主循环：每次处理 4 个配对（即 8 个分量）
        for (; i + 4 <= half; i += 4) {
          // vld1q_f32: 从内存加载 4 个连续 float 到 128-bit NEON 寄存器
          const float32x4_t x0 = vld1q_f32(x + i);       // 前半部分的 4 个分量 x[i..i+3]
          const float32x4_t x1 = vld1q_f32(x + i + half); // 后半部分的 4 个搭档 x[i+half..i+half+3]
          const float32x4_t c = vld1q_f32(cs + i);        // 4 个 cos 值
          const float32x4_t s = vld1q_f32(sn + i);        // 4 个 sin 值

          // 二维旋转公式（乘加分离，贴近 ref 的逐元素舍入行为）：
          // x0' = x0*c - x1*s
          // vmulq_f32: 4 lane 并行乘法
          // vsubq_f32: 4 lane 并行减法
          const float32x4_t r0 = vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s));
          // x1' = x1*c + x0*s
          // vaddq_f32: 4 lane 并行加法
          const float32x4_t r1 = vaddq_f32(vmulq_f32(x1, c), vmulq_f32(x0, s));

          // vst1q_f32: 将 4 个旋转结果写回内存
          vst1q_f32(x + i, r0);         // 写回前半部分
          vst1q_f32(x + i + half, r1);  // 写回后半部分
        }
        // 标量尾段：处理不足 4 个的剩余配对
        for (; i < half; ++i) {
          const float x0 = x[i];           // 前半第 i 个分量
          const float x1 = x[i + half];    // 后半第 i 个搭档
          const float c = cs[i];           // cos 值
          const float s = sn[i];           // sin 值
          x[i] = x0 * c - x1 * s;         // 旋转后的前半
          x[i + half] = x1 * c + x0 * s;  // 旋转后的后半
        }
      };

      // ---- 对 q 的每个 head 施加旋转 ----
      for (int h = 0; h < n_heads; ++h) {
        apply(q + static_cast<size_t>(h) * head_dim); // q[h] 的起始地址
      }
      // ---- 对 k 的每个 head 施加旋转 ----
      // 注意：v 不需要旋转
      for (int h = 0; h < n_kv_heads; ++h) {
        apply(k + static_cast<size_t>(h) * head_dim); // k[h] 的起始地址
      }
    }
  } // namespace

  // 自注册进 rope 注册表，名称为 "neon"。仅在 aarch64 构建中存在。
  // TINYQWEN_ROPE_VARIANT 宏会在 dispatch 表中注册此函数指针，
  // 运行时可通过 backend 选择机制自动选用 NEON 优化版本。
  TINYQWEN_ROPE_VARIANT(rope_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
