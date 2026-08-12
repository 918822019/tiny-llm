#pragma once

#include <cstdint>
#include <string>

#include "tiny_format.h"

namespace tinyqwen {

// 已加载模型文件中某个 tensor 数据的非拥有视图。
// data 指针在所属 ModelFile 的生命周期内一直有效。
// 刻意保持"哑"：只有指针 + shape + dtype，没有引用计数、没有图。
struct TensorView {
  std::string name;
  Dtype dtype = Dtype::kF32;
  int ndim = 0;
  uint64_t shape[4] = {0, 0, 0, 0};
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;

  uint64_t numel() const;
  uint64_t dim(int i) const { return (i >= 0 && i < ndim) ? shape[i] : 1; }

  // 按类型取指针；dtype 不匹配直接 abort（属于程序错误，要大声失败）。
  const float* f32() const;
};

}  // namespace tinyqwen
