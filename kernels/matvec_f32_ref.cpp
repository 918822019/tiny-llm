// 矩阵乘向量：y = W @ x。这是神经网络里最基础也最耗时的运算。
//
// 直观理解：W 有 out_dim 行，每一行和向量 x 做点积，得到 y 的一个分量。
// 模型里所有"投影"（q_proj、up_proj、lm_head……）本质都是这个运算。
//
// 布局：W 是 HF 的行主序 [out_dim, in_dim]，导出时不转置。行主序意味着
// 同一行的元素在内存里是连续的，因此每个输出分量都是一次"连续内存的
// 行点积"，对缓存友好。
//
// 性能地位：这是 decode 阶段的绝对热点——每生成一个 token，绝大部分时间
// 都花在跑很多很多个 matvec 上。后续 INT4/KronQ 量化 kernel 会替换这个
// 入口，但保持相同签名（输入输出含义不变）。
//
// 注意：这里不加 bias；attention 的 q/k/v bias 由调用方在 RoPE 之前加。

#include "ref_ops.h"

#include <cstddef>

namespace tinyqwen {

void matvec_f32_ref(const float* w, const float* x, float* y, int out_dim, int in_dim) {
  // 外层循环：遍历 W 的每一行，算出 y 的一个分量。
  for (int o = 0; o < out_dim; ++o) {
    const float* row = w + static_cast<size_t>(o) * in_dim;  // 第 o 行的起点
    // 用 double 累加：in_dim 可能上千，float 连加会累积舍入误差，
    // double 能让结果更贴近 PyTorch fp32 参考实现（数值对齐很重要）。
    double acc = 0.0;
    for (int i = 0; i < in_dim; ++i) acc += static_cast<double>(row[i]) * x[i];
    y[o] = static_cast<float>(acc);
  }
}

}  // namespace tinyqwen
