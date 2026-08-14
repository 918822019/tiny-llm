#pragma once

// 算子分发层：model 只调用这里的"通用入口"，由分发层决定用哪个实现。
// 这是"保留 base + 可插拔优化"的关键接缝。
//
//   model ──> matvec_f32(通用入口) ──dispatch──> matvec_f32_ref / _double_2_float / ...
//
// 实现采用**自注册**：每个变体在自己的 .cpp 末尾用 TINYQWEN_MATVEC_VARIANT
// 一行宏登记，加变体不需要改 dispatch/main/conf。
// 详见 docs/optimization.md。

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

    // ---- 成对 matvec：y1 = W1 @ x，y2 = W2 @ x（同一个 x）----
    // Qwen 的 k_proj/v_proj 正是这个形状：两个小矩阵共享同一个输入向量。
    // 分开调用时它们各自太小（0.45MB），够不着多线程阈值，只能单线程内联；
    // 合并成一次调用后总量翻倍，有机会摊薄同步开销、走并行路径。
    //
    // 通用入口语义：当前 impl 注册过 pair 实现就用它；没注册则**兜底为调用
    // 两次 matvec_f32**——与调用方分开调数值完全一致，不关心此优化的 impl
    // （ref 等）无需注册任何东西，行为不变。
    using MatvecPairFn = void (*)(const float *w1, const float *w2, const float *x,
                                  float *y1, float *y2, int out_dim, int in_dim);

    void register_matvec_pair_impl(const char *name, MatvecPairFn fn);

    void matvec_pair_f32(const float *w1, const float *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim);

    // ---- f16 权重路径（weight-only 半精度：权重 f16，激活/计算 f32）----
    //
    // f16 实现有**独立注册表**，实现名与 f32 注册表共享同一命名空间
    // （"ref" / "neon_mt_kv_nt" / ...）：选哪个实现由"模型文件的 dtype +
    // 实现名"共同决定（main 按模型 dtype 查对应注册表，未知即报错），
    // forward 按 dtype 调对应入口。两表独立保证"f32 模型配 f16 实现名"
    // 会 fail fast，而不是静默兜底。
    using MatvecF16Fn = void (*)(const uint16_t *w, const float *x, float *y, int out_dim,
                                 int in_dim);

    void register_matvec_f16_impl(const char *name, MatvecF16Fn fn);
    bool set_matvec_f16_impl_by_name(const char *name);
    const char *matvec_f16_impl_name();
    const char *available_matvec_f16_impls();

    // 通用入口：未显式选择时兜底到 f16 注册表里的 "ref"。
    void matvec_f16(const uint16_t *w, const float *x, float *y, int out_dim, int in_dim);

    // f16 成对入口：语义与 matvec_pair_f32 相同；未注册 pair 的 impl 兜底为
    // 调两次 matvec_f16（数值不变）。
    using MatvecPairF16Fn = void (*)(const uint16_t *w1, const uint16_t *w2, const float *x,
                                     float *y1, float *y2, int out_dim, int in_dim);

    void register_matvec_f16_pair_impl(const char *name, MatvecPairF16Fn fn);

    void matvec_pair_f16(const uint16_t *w1, const uint16_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim);

    // ---- qkv 三路融合：q + k + v 共享输入向量，一次 fork-join ----
    // q_dim 和 kv_dim 可以不同（Qwen: 896 vs 128）。
    // 兜底：matvec(q) + matvec_pair(k,v)。
    using MatvecQkvFn = void (*)(const float *wq, const float *wk, const float *wv,
                                 const float *x, float *yq, float *yk, float *yv,
                                 int q_dim, int kv_dim, int in_dim);
    void register_matvec_qkv_impl(const char *name, MatvecQkvFn fn);
    void matvec_qkv_f32(const float *wq, const float *wk, const float *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim);

    using MatvecQkvF16Fn = void (*)(const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
                                    const float *x, float *yq, float *yk, float *yv,
                                    int q_dim, int kv_dim, int in_dim);
    void register_matvec_qkv_f16_impl(const char *name, MatvecQkvF16Fn fn);
    void matvec_qkv_f16(const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim);

    // ---- 非 matvec 算子分发（ops dispatch）----
    //
    // rmsnorm / rope / attention_decode / swiglu / argmax 这五个算子在
    // forward_token 里原来直调 *_ref。这里给它们各建一个小注册表，共享同一个
    // "当前实现名"（默认 "ref"）。通用入口按当前名查自己的注册表，**未注册就
    // 兜底直调对应 _ref**——与 matvec_pair 的兜底语义同款：ref 等实现零改动、
    // 行为不变，优化变体（NEON）自注册后才生效。
    //
    // 函数指针签名与 ref_ops.h 里的 _ref 完全一致（argmax 返回 int）。
    using RmsnormFn = void (*)(const float *x, const float *weight, float *y, int n, float eps);
    using RopeFn = void (*)(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                            int pos, float theta);
    using AttentionDecodeFn = void (*)(const float *q, const float *k_cache, const float *v_cache,
                                       int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                                       int head_dim, float scale, float *out);
    using SwigluFn = void (*)(float *gate, const float *up, int n);
    using ArgmaxFn = int (*)(const float *logits, int n);

    void register_rmsnorm_impl(const char *name, RmsnormFn fn);
    void register_rope_impl(const char *name, RopeFn fn);
    void register_attention_decode_impl(const char *name, AttentionDecodeFn fn);
    void register_swiglu_impl(const char *name, SwigluFn fn);
    void register_argmax_impl(const char *name, ArgmaxFn fn);

    // 按名字选择 ops 实现（五个算子共用一个名字）。找到任一注册即返回 true。
    bool set_ops_impl_by_name(const char *name);
    const char *ops_impl_name();

    // 通用入口：model 调这些而不是直调 _ref。未注册的算子自动兜底到 _ref。
    void rmsnorm(const float *x, const float *weight, float *y, int n, float eps);
    void rope(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta);
    void attention_decode(const float *q, const float *k_cache, const float *v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float *out);
    void swiglu(float *gate, const float *up, int n);
    int argmax(const float *logits, int n);
} // namespace tinyqwen

// 变体自注册宏：写在实现文件末尾、namespace tinyqwen 内部（fn 要用非限定名）。
// 静态 bool 注册器在 main 之前运行（与 tests/test_framework.h 的
// tinytest::Registrar 同款模式）。注意：所在 .cpp 必须真正被链接进来
// （kernels 是 OBJECT 库，所有 .o 都会链入，见 kernels/CMakeLists.txt）。
#define TINYQWEN_MATVEC_VARIANT(fn, name)                                            \
    [[maybe_unused]] static const bool tqwen_reg_##fn =                              \
            (tinyqwen::register_matvec_impl(name, fn), true)

// pair 实现的自注册宏（同上，登记进 pair 注册表；key 与 matvec 实现同名，
// matvec_pair_f32 按当前 impl 名查表）。
#define TINYQWEN_MATVEC_PAIR_VARIANT(fn, name)                                       \
    [[maybe_unused]] static const bool tqwen_reg_pair_##fn =                         \
            (tinyqwen::register_matvec_pair_impl(name, fn), true)

// f16 路径的同款自注册宏（登记进 f16 / f16-pair 注册表）。
#define TINYQWEN_MATVEC_F16_VARIANT(fn, name)                                        \
    [[maybe_unused]] static const bool tqwen_reg_f16_##fn =                          \
            (tinyqwen::register_matvec_f16_impl(name, fn), true)

#define TINYQWEN_MATVEC_F16_PAIR_VARIANT(fn, name)                                   \
    [[maybe_unused]] static const bool tqwen_reg_f16_pair_##fn =                     \
            (tinyqwen::register_matvec_f16_pair_impl(name, fn), true)

// qkv 三路融合的自注册宏。
#define TINYQWEN_MATVEC_QKV_VARIANT(fn, name)                                        \
    [[maybe_unused]] static const bool tqwen_reg_qkv_##fn =                          \
            (tinyqwen::register_matvec_qkv_impl(name, fn), true)

#define TINYQWEN_MATVEC_QKV_F16_VARIANT(fn, name)                                    \
    [[maybe_unused]] static const bool tqwen_reg_qkv_f16_##fn =                      \
            (tinyqwen::register_matvec_qkv_f16_impl(name, fn), true)

// 非 matvec 算子的自注册宏（各登记进对应算子注册表；key 用同一个实现名）。
#define TINYQWEN_RMSNORM_VARIANT(fn, name)                                           \
    [[maybe_unused]] static const bool tqwen_reg_rmsnorm_##fn =                      \
            (tinyqwen::register_rmsnorm_impl(name, fn), true)

#define TINYQWEN_ROPE_VARIANT(fn, name)                                              \
    [[maybe_unused]] static const bool tqwen_reg_rope_##fn =                         \
            (tinyqwen::register_rope_impl(name, fn), true)

#define TINYQWEN_ATTENTION_DECODE_VARIANT(fn, name)                                  \
    [[maybe_unused]] static const bool tqwen_reg_attn_##fn =                         \
            (tinyqwen::register_attention_decode_impl(name, fn), true)

#define TINYQWEN_SWIGLU_VARIANT(fn, name)                                            \
    [[maybe_unused]] static const bool tqwen_reg_swiglu_##fn =                       \
            (tinyqwen::register_swiglu_impl(name, fn), true)

#define TINYQWEN_ARGMAX_VARIANT(fn, name)                                            \
    [[maybe_unused]] static const bool tqwen_reg_argmax_##fn =                       \
            (tinyqwen::register_argmax_impl(name, fn), true)
