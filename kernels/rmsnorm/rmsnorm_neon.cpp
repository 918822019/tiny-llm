// ============================================================================
// rmsnorm_neon.cpp — RMSNorm 归一化的 NEON SIMD 优化版（aarch64）
// ============================================================================
// 本文件实现 RMSNorm 的 ARM NEON 向量化版本：
//   y = x / sqrt(mean(x²) + ε) * weight
//
// 数学定义与 rmsnorm_ref.cpp 完全相同。
//
// 在 Transformer 中的位置：
//   与 ref 版相同——input_layernorm、post_attention_layernorm、最终 norm。
//
// ref 与 neon 版本的关系：
//   - ref 用 double 累加平方和（正确性锚点），本版改用 NEON float 累加。
//   - pass 1（平方和）：4 路独立 float32x4 累加器，vfmaq 一条指令完成平方+累加。
//   - pass 2（归一化×weight）：纯 NEON 向量化，vmulq 两步乘法。
//   - mean/sqrt/scale 的标量计算与 ref 一致。
//
// 优化要点：
//   1. pass 1 使用 4 路独立累加器 + 循环展开到每轮 16 元素，隐藏 FMA 延迟。
//   2. pass 2 用 vmulq 做两步乘法（先 x*scale 再 *weight），结合序同 ref。
//   3. 数值差异来源：pass 1 的 float 累加 vs ref 的 double 累加——
//      平方和的舍入差约 ulp 级，传导到 scale 后对 y 的相对扰动 ~1e-7。
//      远小于贪心决策的 logit 间距，golden token 不受影响。
//   4. 单测按容差门禁（rel 1e-5）对齐 ref。
// ============================================================================

#include "dispatch.h" // TINYQWEN_RMSNORM_VARIANT 自注册宏
#include "ref_ops.h"  // 声明 rmsnorm 函数签名

#if defined(__aarch64__) || defined(_M_ARM64) // 仅在 ARM64 平台编译

#include <arm_neon.h> // ARM NEON SIMD intrinsic 头文件

#include <cmath> // std::sqrt

namespace tinyqwen {
  namespace { // 匿名命名空间：内部函数不暴露到外部链接
    // =========================================================================
    // rmsnorm_neon — RMSNorm 归一化的 NEON 实现
    // =========================================================================
    // 功能：对长度为 n 的输入向量做 RMSNorm 归一化（NEON 加速）
    // 参数：同 rmsnorm_ref
    void rmsnorm_neon(const float *x, const float *weight, float *y, int n, float eps) {
      // ======== pass 1：平方和（4 路独立累加器）========
      // 使用 4 个独立的 float32x4 累加器，避免单条依赖链的瓶颈
      float32x4_t s0 = vdupq_n_f32(0.0f); // 累加器 0：处理第 0, 16, 32... 组
      float32x4_t s1 = vdupq_n_f32(0.0f); // 累加器 1：处理第 4, 20, 36... 组
      float32x4_t s2 = vdupq_n_f32(0.0f); // 累加器 2：处理第 8, 24, 40... 组
      float32x4_t s3 = vdupq_n_f32(0.0f); // 累加器 3：处理第 12, 28, 44... 组
      int i = 0;
      // 主循环：每轮处理 16 个元素（4 组 × 4 lane）
      for (; i + 16 <= n; i += 16) {
        // vld1q_f32: 从内存加载 4 个连续 float 到 NEON 寄存器
        const float32x4_t a = vld1q_f32(x + i);      // 第 0-3 个元素
        const float32x4_t b = vld1q_f32(x + i + 4);  // 第 4-7 个元素
        const float32x4_t c = vld1q_f32(x + i + 8);  // 第 8-11 个元素
        const float32x4_t d = vld1q_f32(x + i + 12); // 第 12-15 个元素
        // vfmaq_f32(s, a, a): fused multiply-add，s += a * a（即平方并累加）
        // 每个累加器独立，CPU 可以流水线并行执行多条 vfmaq
        s0 = vfmaq_f32(s0, a, a); // s0 += a²
        s1 = vfmaq_f32(s1, b, b); // s1 += b²
        s2 = vfmaq_f32(s2, c, c); // s2 += c²
        s3 = vfmaq_f32(s3, d, d); // s3 += d²
      }
      // 合并 4 路累加器为 1 个：vaddq_f32 两两相加
      float32x4_t ssum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
      // 处理剩余不足 16 但 >= 4 的元素（每次 4 个）
      for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i); // 加载 4 个元素
        ssum = vfmaq_f32(ssum, v, v);            // ssum += v²
      }
      // vaddvq_f32: 将 4 个 lane 横向求和为单个 float
      float sumsq = vaddvq_f32(ssum);
      // 标量尾段：处理最后不足 4 个的元素
      for (; i < n; ++i) {
        sumsq += x[i] * x[i];
      }

      // ======== 标量公式：mean → +eps → sqrt → 1/rms（与 ref 相同）========
      // 平方和除以 n 得到均值
      const float mean_sq = sumsq / static_cast<float>(n);
      // 加 eps 后开方得到 RMS（eps 在 sqrt 内部，防止全零时除零）
      const float rms = std::sqrt(mean_sq + eps);
      // 融合成缩放系数 scale = 1/RMS
      const float scale = 1.0f / rms;

      // ======== pass 2：y = (x*scale) * weight，向量化 ========
      // 结合序与 ref 一致：先 x*scale 再 *weight（两次独立的 vmulq）
      const float32x4_t vs = vdupq_n_f32(scale); // 广播 scale 到 4 lane
      int j = 0;
      // 向量化循环：每次处理 4 个元素
      for (; j + 4 <= n; j += 4) {
        // 加载输入 x 和权重 weight 各 4 个元素
        const float32x4_t xv = vld1q_f32(x + j);       // x[j..j+3]
        const float32x4_t wv = vld1q_f32(weight + j);  // weight[j..j+3]
        // vmulq_f32: 4 lane 并行乘法，归一化 = x * scale
        const float32x4_t normed = vmulq_f32(xv, vs);
        // 再次 vmulq_f32: normed * weight，然后 vst1q_f32 写回结果
        vst1q_f32(y + j, vmulq_f32(normed, wv));       // y[j..j+3] = normed * weight
      }
      // 标量尾段：处理不足 4 个的剩余元素
      for (; j < n; ++j) {
        y[j] = (x[j] * scale) * weight[j]; // 结合序同 ref：先乘 scale 再乘 weight
      }
    }
  } // namespace

  // 自注册进 rmsnorm 注册表，名称为 "neon"。仅在 aarch64 构建中存在。
  // TINYQWEN_RMSNORM_VARIANT 宏会在 dispatch 表中注册此函数指针，
  // 运行时可通过 backend 选择机制自动选用 NEON 优化版本。
  TINYQWEN_RMSNORM_VARIANT(rmsnorm_neon, "neon");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
