#include "dispatch.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {
    namespace {
        // 全局当前选择。默认参考实现——保证任何时候行为都和 v1 一致、可兜底。
        MatvecImpl g_matvec_impl = MatvecImpl::kRef;
    } // namespace

    // 优化版变体声明。变体只由分发层调用（model 和测试不直接碰），
    // 所以声明放这里而不是 ref_ops.h（那是 reference kernel 的契约）。
    void matvec_f32_double_2_float(const float *w, const float *x, float *y, int out_dim,
                                   int in_dim);

    void set_matvec_impl(MatvecImpl impl) { g_matvec_impl = impl; }
    MatvecImpl matvec_impl() { return g_matvec_impl; }

    const char *matvec_impl_name() {
        switch (g_matvec_impl) {
            case MatvecImpl::kRef: return "ref";
            case MatvecImpl::kDouble2Float: return "double_2_float";
                // case MatvecImpl::kNeon: return "neon";
        }
        return "unknown";
    }

    // 分发：根据当前选择调用对应实现。任何未覆盖的分支都回退到 ref，
    // 保证永远有正确结果（fail-safe）。
    void matvec_f32(const float *w, const float *x, float *y, int out_dim, int in_dim) {
        switch (g_matvec_impl) {
            case MatvecImpl::kRef:
                matvec_f32_ref(w, x, y, out_dim, in_dim);
                return;
            case MatvecImpl::kDouble2Float:
                matvec_f32_double_2_float(w, x, y, out_dim, in_dim);
                return;
                // case MatvecImpl::kNeon:
                //   matvec_f32_neon(w, x, y, out_dim, in_dim);
                //   return;
        }
        // 兜底：未知实现一律用参考实现。
        matvec_f32_ref(w, x, y, out_dim, in_dim);
    }
} // namespace tinyqwen
