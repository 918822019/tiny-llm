// ============================================================================
// matvec_f32_ref.cpp — 矩阵乘向量（fp32 参考实现）：y = W @ x
// ============================================================================
// 神经网络里最核心的运算。本文件是整个 f32 matvec 优化阶梯的起点和锚点。
//
// 直观理解：W 有 out_dim 行，每一行和向量 x 做点积，得到 y 的一个分量。
// 模型里所有"投影"（q_proj、up_proj、lm_head……）本质都是这个运算。
//
// 布局：W 是 HF 的行主序 [out_dim, in_dim]，导出时不转置。行主序意味着
// 同一行的元素在内存里连续，因此每个输出分量都是一次"连续内存的行点积"。
//
// 性能地位：decode 阶段的绝对热点——每生成一个 token，大部分时间都花在
// 跑很多个 matvec 上。后续 INT4/KronQ 量化 kernel 会替换这个入口，但保持
// 相同签名（输入输出含义不变）。
//
// 数值约定：double 累加（最保守路径），in_dim 可能上千，float 连加会累积
// 舍入误差，double 能让结果更贴近 PyTorch fp32 参考实现。
// 所有 f32 优化变体必须与本实现对齐后才算"算对了"。
//
// 注意：这里不加 bias；attention 的 q/k/v bias 由调用方在 RoPE 之前加。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"  // 辅助函数声明

#include <cstddef>    // size_t

namespace tinyqwen {
  // ========================================================================
  // matvec_f32_ref() — fp32 矩阵乘向量参考实现
  // ========================================================================
  // 功能：计算 y[out_dim] = W[out_dim, in_dim] @ x[in_dim]
  // 参数：
  //   w       — 权重矩阵，行主序 [out_dim, in_dim]
  //   x       — 输入向量（长度 in_dim）
  //   y       — 输出向量（长度 out_dim，调用方分配）
  //   out_dim — 输出维度（W 的行数）
  //   in_dim  — 输入维度（W 的列数 = x 的长度）
  // 算法：逐行做点积，用 double 累加保证数值精度。
  // 返回值：无（结果写入 y）
  void matvec_f32_ref(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    // 外层循环：一行一行处理，每一行算出 y 的一个分量
    for (int o = 0; o < out_dim; ++o) {
      // 定位第 o 行的起点：跳过前面 o 行（每行 in_dim 个元素）
      const float *row = w + static_cast<size_t>(o) * in_dim;
      // double 累加器：in_dim 可能上千，float 连加会累积舍入误差，
      // double 能让结果更贴近 PyTorch fp32 参考实现（数值对齐很重要）
      double acc = 0.0;
      // 内层循环：逐元素乘累加
      for (int i = 0; i < in_dim; ++i) {
        const double weight = static_cast<double>(row[i]); // 权重提升为 double
        const double term = weight * x[i]; // 乘上对应输入（double × double）
        acc += term; // 累加进点积（double 加法）
      }
      // 转回 float，写入输出的第 o 个分量
      y[o] = static_cast<float>(acc);
    }
  }

  // 自注册进 dispatch：--matvec-impl ref / 配置文件 matvec_impl = ref 即可选用。
  // ref 是默认实现 + 兜底，这一行永远不删。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_ref, "ref");
} // namespace tinyqwen
