// ============================================================================
// rmsnorm_ref.cpp — RMSNorm 归一化的参考实现（标量版）
// ============================================================================
// 本文件实现 RMSNorm（Root Mean Square Layer Normalization）的 Qwen 变体。
//
// 数学定义：
//   y = x / sqrt(mean(x²) + ε) * weight
//   其中 mean(x²) = (1/n) * Σᵢ xᵢ²，ε 是防止除零的小常数（通常 1e-6），
//   weight 是每个维度的可学习缩放系数（只有 weight，没有 bias）。
//
// 它是 LayerNorm 的简化版：去掉了均值中心化（减去均值）和 bias 项，
// 只保留"除以均方根"的幅度归一化。好处是计算更快、参数更少，
// 且在实践中效果与完整 LayerNorm 相当。
//
// 在 Transformer 中的位置：
//   - input_layernorm：每个 decoder layer 入口处，attention 之前
//   - post_attention_layernorm：attention 之后、FFN 之前
//   - norm：最后一层输出后、lm_head 之前
//   Qwen2.5-0.5B 共有 24 层 × 2 + 1 = 49 次 RMSNorm 调用/step。
//
// ref 与 neon 版本的关系：
//   - 本文件是标量参考实现，用 double 累加平方和以保证精度（正确性锚点）。
//   - rmsnorm_neon.cpp 改用 NEON float 累加（pass 1 向量化）+ NEON pass 2，
//     数值差异 ~1e-7，单测按容差门禁（rel 1e-5）对齐。
//
// 优化要点：
//   1. 两遍扫描：第一遍求平方和，第二遍归一化并乘 weight。
//   2. 使用 double 累加减少舍入误差（n=896 或 1024 次浮点加法）。
//   3. 将 1/sqrt(...) 融合为一次除法得到 scale，第二遍只做乘法。
//   4. ε 在 sqrt 内部（HuggingFace 的定义），而非 sqrt 外部。
// ============================================================================

#include "ref_ops.h" // 声明 rmsnorm_ref 等参考算子的头文件

#include <cmath> // std::sqrt

namespace tinyqwen {
  // =========================================================================
  // rmsnorm_ref — RMSNorm 归一化参考实现
  // =========================================================================
  // 功能：对长度为 n 的输入向量做 RMSNorm 归一化：y = x / sqrt(mean(x²) + ε) * weight
  // 参数：
  //   x      — 输入向量，长度 n
  //   weight — 可学习的缩放系数，长度 n（每个维度一个）
  //   y      — 输出向量，长度 n
  //   n      — 向量长度（hidden_size 或 inter_size）
  //   eps    — 数值稳定性常数，防止全零向量时除零（Qwen 默认 1e-6）
  void rmsnorm_ref(const float *x, const float *weight, float *y, int n, float eps) {
    // ---- 第 1 遍：求平方和 Σ xᵢ² ----
    // 使用 double 累加减少浮点舍入误差
    // （n 可达 1024，float 累加的相对误差约 n·ε_machine ≈ 1e-4，不可接受）
    double sumsq = 0.0;
    for (int i = 0; i < n; ++i) {
      // 将 x[i] 提升到 double 精度后再平方并累加
      const double xi = static_cast<double>(x[i]);
      sumsq += xi * xi;
    }

    // 平方和取平均，得到 mean(x²)
    const float mean_sq = static_cast<float>(sumsq / n);

    // 加 eps 后开方得到 RMS = sqrt(mean(x²) + ε)
    // 注意：eps 在 sqrt 内部（这是 HuggingFace transformers 库的定义）
    // 当 x 全为零时，RMS = sqrt(ε) > 0，避免除零
    const float rms = std::sqrt(mean_sq + eps);

    // 融合成缩放系数 scale = 1 / RMS
    // 后续只需做一次乘法即可完成归一化，避免每次循环都做除法
    const float scale = 1.0f / rms;

    // ---- 第 2 遍：归一化并乘以 weight ----
    for (int i = 0; i < n; ++i) {
      // 先归一化：x[i] * scale = x[i] / RMS
      const float normed = x[i] * scale;
      // 再乘以对应维度的可学习缩放系数
      y[i] = normed * weight[i];
    }
  }
} // namespace tinyqwen
