// ============================================================================
// matvec_f32_double_2_float.cpp — 矩阵乘向量：y = W @ x —— float 累加版
// ============================================================================
// 归因阶梯上的位置（每层只改一个变量，A/B 才能说清功劳归谁）：
//
//   ref（标量、double 累加）
//     → double_2_float（本文件） [+ float 累加]   ← 隔离"累加精度"
//     → acc4                   [+ 4 条独立累加链]
//     → neon_nofma             [+ NEON 向量化]
//     → neon                   [+ FMA 融合]
//
// 本文件在 ref 的基础上只改一件事：把 double 累加换成 float 累加。
// 目的是单独量化"累加精度从 double 降到 float"对性能的影响——
// float 加法比 double 快（在某些微架构上吞吐更高），但舍入误差更大。
//
// 数学含义与 ref 完全一致：y[o] = Σ_i W[o,i] * x[i]。
// 数值差异：float 累加的舍入链更长（in_dim 次 float 加法 vs double），
// 误差比 ref 大但仍可接受，单测容差内对齐。
//
// 选用：--matvec-impl double_2_float / tinyqwen.conf 里 matvec_impl = double_2_float。
// 注意：这里不加 bias；attention 的 q/k/v bias 由调用方在 RoPE 之前加。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"  // 辅助函数声明

#include <cstddef>    // size_t

namespace tinyqwen {
  // ========================================================================
  // matvec_f32_double_2_float() — fp32 矩阵乘向量（float 累加版）
  // ========================================================================
  // 功能：计算 y[out_dim] = W[out_dim, in_dim] @ x[in_dim]
  // 参数：
  //   w       — 权重矩阵，行主序 [out_dim, in_dim]
  //   x       — 输入向量（长度 in_dim）
  //   y       — 输出向量（长度 out_dim，调用方分配）
  //   out_dim — 输出维度（W 的行数）
  //   in_dim  — 输入维度（W 的列数 = x 的长度）
  // 算法：逐行做点积，用 float（而非 ref 的 double）累加。
  // 返回值：无（结果写入 y）
  void matvec_f32_double_2_float(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    // 外层循环：一行一行处理，每一行算出 y 的一个分量
    for (int o = 0; o < out_dim; ++o) {
      // 定位第 o 行的起点：跳过前面 o 行（每行 in_dim 个元素）
      const float *row = w + static_cast<size_t>(o) * in_dim;
      // float 累加器（区别于 ref 的 double 累加器）
      float acc = 0.0;
      // 内层循环：逐元素乘累加
      for (int i = 0; i < in_dim; ++i) {
        const float term = row[i] * x[i]; // 权重 × 输入（fp32 乘法）
        acc += term; // 累加进点积（fp32 加法，区别于 ref 的 fp64 加法）
      }
      // 直接写入 float 结果（无需类型转换，因为累加器本身就是 float）
      y[o] = acc;
    }
  }

  // 自注册进 dispatch：--matvec-impl double_2_float / 配置文件可选
  TINYQWEN_MATVEC_VARIANT(matvec_f32_double_2_float, "double_2_float");
} // namespace tinyqwen
