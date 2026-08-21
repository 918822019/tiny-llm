// ============================================================================
// matvec_f32_acc4.cpp — 矩阵乘向量：y = W @ x —— 标量 4 链累加版
// ============================================================================
// 归因阶梯上的位置（每层只加一个技术，A/B 才能说清功劳归谁）：
//
//   ref（标量、double 累加）
//     → double_2_float      [+ float 累加]
//     → acc4（本文件）       [+ 4 条独立累加链]   ← 隔离"累加结构"
//     → neon_nofma          [+ NEON 向量化，不用 FMA]
//     → neon                [+ FMA 融合]
//
// 本文件只隔离"多链累加"这一个技术：
//   - 仍然标量：不用任何 SIMD 向量指令，一次还是处理 1 个元素；
//   - 累加精度不变：和 double_2_float 一样是 float 累加；
//   - 唯一的变化：把"一条串行依赖的累加链"拆成"4 条互不依赖的链"。
//
// 为什么这样也能快：浮点乘加有 ~3-4 个周期的延迟。单链写法
// acc = acc + a*b 每一步都依赖上一步的结果，流水线只能干等；
// 4 条独立链让乱序执行把延迟重叠起来（咖啡机类比详见
// matvec_f32_neon.cpp 文件头"为什么要 4 个累加器"）。
// 这个技术与 SIMD 无关——标量代码照样能用，本文件就是单独证明它。
//
// 数值说明：float 累加，误差量级与 double_2_float 相同；4 链的合并
// 顺序与 ref 的串行不同（浮点加法不满足结合律），差异由单测容差处理。
// 选用：--matvec-impl acc4 / tinyqwen.conf 里 matvec_impl = acc4。
// 注意：这里不加 bias；与 ref 相同。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"  // 辅助函数声明

#include <cstddef>    // size_t

namespace tinyqwen {
  namespace {
    // ========================================================================
    // dot_row_acc4() — W 的一行与 x 的点积：标量、4 条独立累加链
    // ========================================================================
    // 功能：计算 row[0..n-1] 与 x[0..n-1] 的点积
    // 参数：
    //   row — 权重行的起始地址（fp32）
    //   x   — 输入向量（fp32）
    //   n   — 向量长度（in_dim）
    // 返回值：点积结果（fp32）
    // 说明：纯标量实现，不使用任何 SIMD 指令。唯一的优化是将单条累加链
    //       拆成 4 条互不依赖的链，利用 CPU 乱序执行重叠浮点运算延迟。
    inline float dot_row_acc4(const float *row, const float *x, int n) {
      // 4 个独立累加器：谁也不依赖谁的结果，CPU 可以同时推进
      // （对比单链 acc += ... 每条都依赖上一条的结果，FMA 延迟无法重叠）
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      float acc2 = 0.0f;
      float acc3 = 0.0f;

      int i = 0;
      // 主循环：一次迭代吃 4 个元素，每条链分 1 个
      // n4 = n 向下取整到 4 的倍数（抹掉低 2 位），保证 i+3 不越界
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        // 四条语句互不依赖（各写各的 acc）——这就是"多链"的全部秘密
        // （-O3 下编译器可能把每条融合成一条标量 FMA 指令；ref 和
        //  double_2_float 同样会被融合，所以这不构成本层的变量）
        acc0 += row[i] * x[i];         // 链 0：处理第 i 个元素
        acc1 += row[i + 1] * x[i + 1]; // 链 1：处理第 i+1 个元素
        acc2 += row[i + 2] * x[i + 2]; // 链 2：处理第 i+2 个元素
        acc3 += row[i + 3] * x[i + 3]; // 链 3：处理第 i+3 个元素
      }
      // 合并 4 条链的部分和（顺序与 ref 串行不同，舍入差异极小）
      // (acc0+acc1)+(acc2+acc3) 形成平衡二叉归约树，比串行累加误差更小
      float total = (acc0 + acc1) + (acc2 + acc3);
      // 标量尾段：还剩 0~3 个元素，逐个补上（不会越界读）
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }
  } // namespace

  // ========================================================================
  // matvec_f32_acc4() — 外层入口
  // ========================================================================
  // 功能：计算 y = W @ x（fp32，标量 4 链累加）
  // 参数：
  //   w       — 权重矩阵，行主序 [out_dim, in_dim]
  //   x       — 输入向量（长度 in_dim）
  //   y       — 输出向量（长度 out_dim）
  //   out_dim — 输出维度
  //   in_dim  — 输入维度
  // 说明：结构与 ref 完全一致：每行一个点积。变体只改点积的实现。
  void matvec_f32_acc4(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; ++o) {
      // 定位第 o 行的起点
      const float *row = w + static_cast<size_t>(o) * in_dim;
      // 用 4 链累加点积计算该行的输出分量
      y[o] = dot_row_acc4(row, x, in_dim);
    }
  }

  // 自注册进 dispatch：--matvec-impl acc4 即可选用（所有平台可用，纯标量）
  TINYQWEN_MATVEC_VARIANT(matvec_f32_acc4, "acc4");
} // namespace tinyqwen
