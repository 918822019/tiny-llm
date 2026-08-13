// 矩阵乘向量：y = W @ x。神经网络里最核心的运算。
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
// 注意：这里不加 bias；attention 的 q/k/v bias 由调用方在 RoPE 之前加。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>

namespace tinyqwen {
  void matvec_f32_ref(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    // 外层循环：一行一行处理，每一行算出 y 的一个分量。
    for (int o = 0; o < out_dim; ++o) {
      // 定位第 o 行的起点：跳过前面 o 行（每行 in_dim 个元素）。
      const float *row = w + static_cast<size_t>(o) * in_dim;
      // 第 o 行与 x 做点积。
      // 用 double 累加：in_dim 可能上千，float 连加会累积舍入误差，
      // double 能让结果更贴近 PyTorch fp32 参考实现（数值对齐很重要）。
      double acc = 0.0;
      for (int i = 0; i < in_dim; ++i) {
        const double weight = static_cast<double>(row[i]); // 权重提升为 double
        const double term = weight * x[i]; // 乘上对应输入
        acc += term; // 累加进点积
      }
      // 转回 float，写入输出的第 o 个分量。
      y[o] = static_cast<float>(acc);
    }
  }

  // 自注册进 dispatch：--matvec-impl ref / 配置文件 matvec_impl = ref 即可选用。
  // ref 是默认实现 + 兜底，这一行永远不删。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_ref, "ref");
} // namespace tinyqwen
