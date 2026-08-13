#pragma once

// 算子分发层：model 只调用这里的"通用入口"，由分发层决定用哪个实现。
// 这是"保留 base + 可插拔优化"的关键接缝。
//
//   model ──> matvec_f32(通用入口) ──dispatch──> matvec_f32_ref / _double_2_float / ...
//
// 实现采用**自注册**：每个变体在自己的 .cpp 末尾用 TINYQWEN_MATVEC_VARIANT
// 一行宏登记，加变体不需要改 dispatch/main/conf。
// 详见 docs/kernel_optimization.md。

#include "ref_ops.h"

namespace tinyqwen {
    // matvec 实现的统一签名（与 matvec_f32_ref 一致）。
    using MatvecFn = void (*)(const float *w, const float *x, float *y, int out_dim, int in_dim);

    // 注册一个实现（通常不直接调用，而是用文件末尾的 TINYQWEN_MATVEC_VARIANT 宏）。
    void register_matvec_impl(const char *name, MatvecFn fn);

    // 按名字选择实现。找到返回 true；未找到返回 false 且不改变当前选择。
    bool set_matvec_impl_by_name(const char *name);

    // 当前实现的名字（未显式选择时为 "ref"——matvec_f32 会兜底到 ref）。
    const char *matvec_impl_name();

    // 所有已注册实现名，逗号分隔（报错/帮助用）。
    const char *available_matvec_impls();

    // 通用入口：model 调用这个，而不是直接调某个具体实现。
    // 未显式选择时自动用 "ref"；ref 都没注册（链接配置错误）则 abort。
    void matvec_f32(const float *w, const float *x, float *y, int out_dim, int in_dim);
} // namespace tinyqwen

// 变体自注册宏：写在实现文件末尾、namespace tinyqwen 内部（fn 要用非限定名）。
// 静态 bool 注册器在 main 之前运行（与 tests/test_framework.h 的
// tinytest::Registrar 同款模式）。注意：所在 .cpp 必须真正被链接进来
// （kernels 是 OBJECT 库，所有 .o 都会链入，见 kernels/CMakeLists.txt）。
#define TINYQWEN_MATVEC_VARIANT(fn, name)                                            \
    [[maybe_unused]] static const bool tqwen_reg_##fn =                              \
            (tinyqwen::register_matvec_impl(name, fn), true)
