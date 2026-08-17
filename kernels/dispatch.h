#pragma once

// 算子分发层：model 只调用这里的"通用入口"，由分发层决定用哪个实现。
// 这是"保留 base + 可插拔优化"的关键接缝。
//
//   model ──> matvec_f32(通用入口) ──dispatch──> matvec_f32_ref / _double_2_float / ...
//
// 实现采用**自注册**：每个变体在自己的 .cpp 末尾用 TINYQWEN_MATVEC_VARIANT
// 一行宏登记，加变体不需要改 dispatch/main/conf。
// 详见 docs/optimization.md。

#include <string>

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

    // ---- INT4 weight-only 路径（非对称 uint4，per-group scale+zero，interleaved）----
    //
    // 签名比 f32/f16 多一个 group_size 参数：kernel 内部按 interleaved 布局
    // 逐组解包反量化。权重指针类型 = const uint8_t*（packed bytes，包含 scale/zero）。
    using MatvecI4Fn = void (*)(const uint8_t *w, const float *x, float *y,
                                int out_dim, int in_dim, int group_size);

    void register_matvec_i4_impl(const char *name, MatvecI4Fn fn);
    bool set_matvec_i4_impl_by_name(const char *name);
    const char *matvec_i4_impl_name();
    const char *available_matvec_i4_impls();
    void matvec_i4(const uint8_t *w, const float *x, float *y,
                   int out_dim, int in_dim, int group_size);

    using MatvecPairI4Fn = void (*)(const uint8_t *w1, const uint8_t *w2, const float *x,
                                    float *y1, float *y2, int out_dim, int in_dim, int group_size);
    void register_matvec_i4_pair_impl(const char *name, MatvecPairI4Fn fn);
    void matvec_pair_i4(const uint8_t *w1, const uint8_t *w2, const float *x,
                        float *y1, float *y2, int out_dim, int in_dim, int group_size);

    using MatvecQkvI4Fn = void (*)(const uint8_t *wq, const uint8_t *wk, const uint8_t *wv,
                                   const float *x, float *yq, float *yk, float *yv,
                                   int q_dim, int kv_dim, int in_dim, int group_size);
    void register_matvec_qkv_i4_impl(const char *name, MatvecQkvI4Fn fn);
    void matvec_qkv_i4(const uint8_t *wq, const uint8_t *wk, const uint8_t *wv,
                       const float *x, float *yq, float *yk, float *yv,
                       int q_dim, int kv_dim, int in_dim, int group_size);

    // ---- Matmul (GEMM) 路径：prefill 批量投影 ----
    //
    // Y[M,N] = W[M,K] × X[K,N]。X 和 Y 按列主序存储：每列 = 一个 token 的向量。
    // W 行主序（与 matvec 共用同一份权重布局）。
    // N=1 时退化为 matvec，但专门的 matvec kernel 通常更快（少 loop overhead）。
    using MatmulFn = void (*)(const float *w, const float *x, float *y,
                              int M, int K, int N);
    void register_matmul_impl(const char *name, MatmulFn fn);
    bool set_matmul_impl_by_name(const char *name);
    void matmul_f32(const float *w, const float *x, float *y, int M, int K, int N);

    using MatmulI4Fn = void (*)(const uint8_t *w, const float *x, float *y,
                                int M, int K, int N, int group_size);
    void register_matmul_i4_impl(const char *name, MatmulI4Fn fn);
    void matmul_i4(const uint8_t *w, const float *x, float *y,
                   int M, int K, int N, int group_size);

#define TINYQWEN_MATMUL_VARIANT(fn, name)                                              \
    [[maybe_unused]] static const bool tqwen_reg_mm_##fn =                             \
            (tinyqwen::register_matmul_impl(name, fn), true)

#define TINYQWEN_MATMUL_I4_VARIANT(fn, name)                                           \
    [[maybe_unused]] static const bool tqwen_reg_mm_i4_##fn =                          \
            (tinyqwen::register_matmul_i4_impl(name, fn), true)

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

    // 按名字选择 ops 实现（所有非 matvec 算子共用一个名字）。找到任一注册即返回 true。
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

    // ---- GPU decode engine 分发 ----
    //
    // 与上面"逐算子"分发不同：这是一个"整段 forward"级的可插拔入口。engine
    // 把权重/激活/KV cache 全部常驻显存，decode_step 在单条 CUDA stream 上
    // 串起整个前向，只有 token id 过 PCIe——消灭逐 matvec 的 CPU↔GPU 桥接。
    //
    // GpuDecodeEngine 是**不透明句柄**：这里只有前向声明，完整定义在 CUDA
    // engine 的翻译单元里。runtime/ 只经由本头文件拿到指针、原样传回，
    // 永远不解引用——因此 runtime 保持 CUDA-agnostic（不 include 任何 CUDA
    // 头）。CPU forward 永远是兜底参考：没有 engine 注册/选中时一切照旧。
    //
    // model_file 以 const void* 传递（实为 const ModelFile*），避免本头文件
    // 依赖 runtime 的 model_loader.h；engine 的 .cu 里再 cast 回去。
    struct GpuDecodeEngine; // opaque

    using GpuDecodeCreateFn = bool (*)(const void *model_file, int max_seq_len,
                                       std::string *err, GpuDecodeEngine **out);
    using GpuDecodeStepFn = int (*)(GpuDecodeEngine *e, int token_id);
    using GpuDecodeResetFn = void (*)(GpuDecodeEngine *e);
    using GpuDecodeDestroyFn = void (*)(GpuDecodeEngine *e);
    using GpuDecodeLogitsFn = const float *(*)(const GpuDecodeEngine *e);

    void register_gpu_decode_impl(const char *name, GpuDecodeCreateFn create,
                                  GpuDecodeStepFn step, GpuDecodeResetFn reset,
                                  GpuDecodeDestroyFn destroy, GpuDecodeLogitsFn logits);
    bool set_gpu_decode_impl_by_name(const char *name);
    const char *gpu_decode_impl_name(); // 未选择时为 ""
    const char *available_gpu_decode_impls();

    // 通用入口。create 在未选择实现时返回 false（调用方回退 CPU forward）。
    bool gpu_decode_available();
    bool gpu_decode_create(const void *model_file, int max_seq_len, std::string *err,
                           GpuDecodeEngine **out);
    int gpu_decode_step(GpuDecodeEngine *e, int token_id);
    void gpu_decode_reset(GpuDecodeEngine *e);
    void gpu_decode_destroy(GpuDecodeEngine *e);
    const float *gpu_decode_logits(const GpuDecodeEngine *e); // device 指针（供 dump）

    // ---- GDN（Gated DeltaNet）算子分发 ----
    //
    // Qwen3.5 的 GDN 层专属算子：l2norm / causal conv1d / gated delta rule 递归 /
    // 门控 RMSNorm。与上面五个算子共用同一个实现名（--ops-impl neon 同时启用）。
    using CausalConv1dUpdateFn = void (*)(const float *x, float *conv_state,
                                          const float *weight, float *out,
                                          int dim, int kernel_size);
    using L2normInplaceFn = void (*)(float *x, int n, float eps);
    using GdnStepFn = void (*)(float *S, const float *q, const float *k, const float *v,
                               float g, float beta, float *o, int qk_dim, int v_dim);
    using RmsnormGatedFn = void (*)(const float *x, const float *gate, const float *weight,
                                    float *y, int n, float eps);

    void register_causal_conv1d_update_impl(const char *name, CausalConv1dUpdateFn fn);
    void register_l2norm_inplace_impl(const char *name, L2normInplaceFn fn);
    void register_gdn_step_impl(const char *name, GdnStepFn fn);
    void register_rmsnorm_gated_impl(const char *name, RmsnormGatedFn fn);

    void causal_conv1d_update(const float *x, float *conv_state, const float *weight,
                              float *out, int dim, int kernel_size);
    void l2norm_inplace(float *x, int n, float eps);
    void gdn_step(float *S, const float *q, const float *k, const float *v,
                  float g, float beta, float *o, int qk_dim, int v_dim);
    void rmsnorm_gated(const float *x, const float *gate, const float *weight,
                       float *y, int n, float eps);
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

// INT4 路径的自注册宏（登记进 i4 / i4-pair / i4-qkv 注册表）。
#define TINYQWEN_MATVEC_I4_VARIANT(fn, name)                                         \
    [[maybe_unused]] static const bool tqwen_reg_i4_##fn =                           \
            (tinyqwen::register_matvec_i4_impl(name, fn), true)

#define TINYQWEN_MATVEC_I4_PAIR_VARIANT(fn, name)                                    \
    [[maybe_unused]] static const bool tqwen_reg_i4_pair_##fn =                      \
            (tinyqwen::register_matvec_i4_pair_impl(name, fn), true)

#define TINYQWEN_MATVEC_QKV_I4_VARIANT(fn, name)                                     \
    [[maybe_unused]] static const bool tqwen_reg_qkv_i4_##fn =                       \
            (tinyqwen::register_matvec_qkv_i4_impl(name, fn), true)

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

// GPU decode engine 的自注册宏：engine 的 .cu 文件末尾调用一次，登记
// create/step/reset/destroy/logits 五个函数指针。仅 CUDA 构建会执行到这里；
// 无 CUDA 时没有 engine 注册，gpu_decode_available() 为 false，自动走 CPU。
#define TINYQWEN_GPU_DECODE_VARIANT(name_str, create_fn, step_fn, reset_fn,          \
                                    destroy_fn, logits_fn)                            \
    [[maybe_unused]] static const bool tqwen_reg_gpud_##create_fn =                  \
            (tinyqwen::register_gpu_decode_impl(name_str, create_fn, step_fn,        \
                                                reset_fn, destroy_fn, logits_fn),     \
             true)

// GDN 算子的自注册宏（共用 ops 实现名）。
#define TINYQWEN_CAUSAL_CONV1D_UPDATE_VARIANT(fn, name)                              \
    [[maybe_unused]] static const bool tqwen_reg_conv1d_##fn =                       \
            (tinyqwen::register_causal_conv1d_update_impl(name, fn), true)

#define TINYQWEN_L2NORM_INPLACE_VARIANT(fn, name)                                    \
    [[maybe_unused]] static const bool tqwen_reg_l2norm_##fn =                       \
            (tinyqwen::register_l2norm_inplace_impl(name, fn), true)

#define TINYQWEN_GDN_STEP_VARIANT(fn, name)                                          \
    [[maybe_unused]] static const bool tqwen_reg_gdn_step_##fn =                     \
            (tinyqwen::register_gdn_step_impl(name, fn), true)

#define TINYQWEN_RMSNORM_GATED_VARIANT(fn, name)                                     \
    [[maybe_unused]] static const bool tqwen_reg_rmsnorm_gated_##fn =                \
            (tinyqwen::register_rmsnorm_gated_impl(name, fn), true)
