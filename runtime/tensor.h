#pragma once

// TensorView：一个"只读视图"，描述某块权重数据在哪里、什么形状。
// 先读 docs/infra_primer.md 第 3 节（内存布局与指针）再看这里。

#include <cstdint>
#include <string>

#include "tiny_format.h"

namespace tinyqwen {

// 一个 tensor 的非拥有（non-owning）视图。
//
// 什么叫"非拥有"？
//   它只记录"数据的地址 + 形状"，自己不负责分配/释放那块内存。
//   真正的内存属于加载了文件的 ModelFile。好处是零拷贝、零开销；
//   代价是：ModelFile 必须比所有用到这个视图的地方活得更久，
//   否则指针就"悬空"了（指向已被释放的内存，读到垃圾）。
//
// 刻意保持"哑"：只有指针 + shape + dtype，没有引用计数、没有计算逻辑。
struct TensorView {
  std::string name;              // tensor 名字（如 "model.layers.0.mlp.up_proj.weight"）
  Dtype dtype = Dtype::kF32;     // 数据类型
  int ndim = 0;                  // 维度数（1..4）
  uint64_t shape[4] = {0, 0, 0, 0};  // 每一维的大小
  const uint8_t* data = nullptr;     // 指向数据的指针（只读）
  uint64_t nbytes = 0;           // 数据总字节数

  // 元素总个数 = shape 各维相乘。
  uint64_t numel() const;

  // 取第 i 维大小；越界时安全地返回 1（相当于"这一维不存在"）。
  uint64_t dim(int i) const { return (i >= 0 && i < ndim) ? shape[i] : 1; }

  // 按 float 类型取数据指针。如果实际 dtype 不是 f32，说明是程序写错了，
  // 直接 abort 大声失败（见 primer 第 9 节 fail fast）。
  const float* f32() const;
};

}  // namespace tinyqwen
