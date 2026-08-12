#include "dispatch.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {

namespace {
// 全局当前选择。默认参考实现——保证任何时候行为都和 v1 一致、可兜底。
MatvecImpl g_matvec_impl = MatvecImpl::kRef;
}  // namespace

void set_matvec_impl(MatvecImpl impl) { g_matvec_impl = impl; }
MatvecImpl matvec_impl() { return g_matvec_impl; }

const char* matvec_impl_name() {
  switch (g_matvec_impl) {
    case MatvecImpl::kRef: return "ref";
    // case MatvecImpl::kNeon: return "neon";
  }
  return "unknown";
}

// 分发：根据当前选择调用对应实现。任何未覆盖的分支都回退到 ref，
// 保证永远有正确结果（fail-safe）。
void matvec_f32(const float* w, const float* x, float* y, int out_dim, int in_dim) {
  switch (g_matvec_impl) {
    case MatvecImpl::kRef:
      matvec_f32_ref(w, x, y, out_dim, in_dim);
      return;
    // case MatvecImpl::kNeon:
    //   matvec_f32_neon(w, x, y, out_dim, in_dim);
    //   return;
  }
  // 兜底：未知实现一律用参考实现。
  matvec_f32_ref(w, x, y, out_dim, in_dim);
}

}  // namespace tinyqwen
