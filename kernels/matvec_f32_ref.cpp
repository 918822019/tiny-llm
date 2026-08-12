// y = W @ x，fp32 权重采用 HF 布局：W 行主序 [out_dim, in_dim]，导出时不转置，
// 因此每个输出分量都是一次连续的行点积。这是 decode 阶段的绝对热点
// （所有 projection + lm_head 都走这里），后续 INT4/KronQ kernel 会替换
// 这个入口，但保持相同签名。
//
// 这里不加 bias；attention 的 q/k/v bias 由调用方（runtime/qwen_model.cpp）
// 在 RoPE 之前加上。

#include "ref_ops.h"

#include <cstddef>

namespace tinyqwen {

void matvec_f32_ref(const float* w, const float* x, float* y, int out_dim, int in_dim) {
  for (int o = 0; o < out_dim; ++o) {
    const float* row = w + static_cast<size_t>(o) * in_dim;
    double acc = 0.0;  // double 累加，贴近 PyTorch fp32 结果
    for (int i = 0; i < in_dim; ++i) acc += static_cast<double>(row[i]) * x[i];
    y[o] = static_cast<float>(acc);
  }
}

}  // namespace tinyqwen
