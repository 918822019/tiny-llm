#pragma once

// 算子分发层：model 只调用这里的"通用入口"，由分发层决定用哪个实现。
// 这是"保留 base + 可插拔优化"的关键接缝。
//
//   model ──> matvec_f32(通用入口) ──dispatch──> matvec_f32_ref / _neon / ...
//
// 详见 docs/kernel_optimization.md。

#include "ref_ops.h"

namespace tinyqwen {

// matvec 的可选实现。ref 永远存在（正确性基准 + 兜底），只增不删。
// 以后新增优化版：在这里加一个枚举值 + 在 dispatch.cpp 加一个 case +
// 新建一个 <op>_<dtype>_<variant>.cpp 实现文件。
enum class MatvecImpl {
  kRef = 0,        // 标量参考实现（默认，行为与 v1 完全一致）
  // kNeon,        // 预留：NEON SIMD
  // kNeonMt,      // 预留：NEON + 多线程
};

// 选择 / 查询当前使用的实现。
void set_matvec_impl(MatvecImpl impl);
MatvecImpl matvec_impl();
const char* matvec_impl_name();

// 通用入口：model 调用这个，而不是直接调某个具体实现。
void matvec_f32(const float* w, const float* x, float* y, int out_dim, int in_dim);

}  // namespace tinyqwen
