#include "tensor.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

size_t dtype_size(Dtype dtype) {
  switch (dtype) {
    case Dtype::kF32: return 4;
    case Dtype::kF16: return 2;
    case Dtype::kI8: return 1;
    case Dtype::kI4: return 0;  // sub-byte; layout is kernel-specific
  }
  return 0;
}

const char* dtype_name(Dtype dtype) {
  switch (dtype) {
    case Dtype::kF32: return "f32";
    case Dtype::kF16: return "f16";
    case Dtype::kI8: return "i8";
    case Dtype::kI4: return "i4";
  }
  return "?";
}

uint64_t TensorView::numel() const {
  uint64_t n = 1;
  for (int i = 0; i < ndim; ++i) n *= shape[i];
  return n;
}

const float* TensorView::f32() const {
  if (dtype != Dtype::kF32) {
    std::fprintf(stderr, "tinyqwen: tensor '%s' has dtype %s, f32 expected\n",
                 name.c_str(), dtype_name(dtype));
    std::abort();
  }
  return reinterpret_cast<const float*>(data);
}

}  // namespace tinyqwen
