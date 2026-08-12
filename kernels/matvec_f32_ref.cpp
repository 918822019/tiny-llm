// y = W @ x for fp32 weights in HF layout: W is row-major [out_dim, in_dim]
// and NOT transposed at export time, so each output is one contiguous
// row-dot-product. This is THE dominant cost of the decode step (every
// projection + lm_head goes through here), which is why INT4/KronQ kernels
// will replace this entry point later while keeping the same signature.
//
// No bias is applied here; attention q/k/v biases are added by the caller
// (runtime/qwen_model.cpp) before RoPE.

#include "ref_ops.h"

#include <cstddef>

namespace tinyqwen {

void matvec_f32_ref(const float* w, const float* x, float* y, int out_dim, int in_dim) {
  for (int o = 0; o < out_dim; ++o) {
    const float* row = w + static_cast<size_t>(o) * in_dim;
    double acc = 0.0;  // double accumulation to stay close to PyTorch fp32
    for (int i = 0; i < in_dim; ++i) acc += static_cast<double>(row[i]) * x[i];
    y[o] = static_cast<float>(acc);
  }
}

}  // namespace tinyqwen
