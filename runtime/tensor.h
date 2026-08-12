#pragma once

#include <cstdint>
#include <string>

#include "tiny_format.h"

namespace tinyqwen {

// Non-owning view of one tensor payload inside a loaded model file.
// The data pointer stays valid for the lifetime of the owning ModelFile.
// Intentionally dumb: pointer + shape + dtype, no refcount, no graph.
struct TensorView {
  std::string name;
  Dtype dtype = Dtype::kF32;
  int ndim = 0;
  uint64_t shape[4] = {0, 0, 0, 0};
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;

  uint64_t numel() const;
  uint64_t dim(int i) const { return (i >= 0 && i < ndim) ? shape[i] : 1; }

  // Typed access; aborts on dtype mismatch (programming error, fail loud).
  const float* f32() const;
};

}  // namespace tinyqwen
