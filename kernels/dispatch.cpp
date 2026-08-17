#include "dispatch.h"
#include "gdn_ops.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyqwen {
    namespace {
        // 名字 -> 实现。Meyers singleton：首次调用时构造，C++11 起线程安全，
        // 保证它一定先于任何 register_matvec_impl 调用存在（无静态初始化顺序坑）。
        std::unordered_map<std::string, MatvecFn> &registry() {
            static std::unordered_map<std::string, MatvecFn> r;
            return r;
        }

        // 当前选择。nullptr = 未显式选择，matvec_f32 会兜底到 "ref"。
        MatvecFn g_current = nullptr;
        std::string g_current_name; // 空 = 未显式选择
    } // namespace

    void register_matvec_impl(const char *name, MatvecFn fn) { registry()[name] = fn; }

    bool set_matvec_impl_by_name(const char *name) {
        const auto &r = registry();
        auto it = r.find(name);
        if (it == r.end()) return false; // 未知名字：不改变当前选择，由调用方报错
        g_current = it->second;
        g_current_name = name;
        return true;
    }

    const char *matvec_impl_name() {
        return g_current_name.empty() ? "ref" : g_current_name.c_str();
    }

    const char *available_matvec_impls() {
        // 注册都发生在 main 之前且之后不变，拼一次缓存起来。排序保证输出稳定。
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    void matvec_f32(const float *w, const float *x, float *y, int out_dim, int in_dim) {
        MatvecFn fn = g_current;
        if (!fn) {
            // 兜底到 ref：只要 ref 实现被链接进来，行为就和 v1 完全一致。
            auto it = registry().find("ref");
            if (it == registry().end()) {
                std::fprintf(stderr,
                             "tinyqwen: matvec 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;
        }
        fn(w, x, y, out_dim, in_dim);
    }

    namespace {
        // pair 实现注册表：key 与 matvec 实现同名。注册同样发生在 main 之前。
        std::unordered_map<std::string, MatvecPairFn> &pair_registry() {
            static std::unordered_map<std::string, MatvecPairFn> r;
            return r;
        }
    } // namespace

    void register_matvec_pair_impl(const char *name, MatvecPairFn fn) {
        pair_registry()[name] = fn;
    }

    void matvec_pair_f32(const float *w1, const float *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        // 当前 impl 注册了 pair 实现就用它（key = impl 名，未显式选择时为 "ref"）。
        const auto &pr = pair_registry();
        auto it = pr.find(matvec_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim);
            return;
        }
        // 兜底：等价于调用方分开调两次 matvec_f32（数值一致，行为不变）。
        matvec_f32(w1, x, y1, out_dim, in_dim);
        matvec_f32(w2, x, y2, out_dim, in_dim);
    }

    // ---- f16 路径：与 f32 完全对称的第二套注册表/选择器/入口 ----
    namespace {
        std::unordered_map<std::string, MatvecF16Fn> &f16_registry() {
            static std::unordered_map<std::string, MatvecF16Fn> r;
            return r;
        }

        std::unordered_map<std::string, MatvecPairF16Fn> &f16_pair_registry() {
            static std::unordered_map<std::string, MatvecPairF16Fn> r;
            return r;
        }

        MatvecF16Fn g_f16_current = nullptr;
        std::string g_f16_current_name;
    } // namespace

    void register_matvec_f16_impl(const char *name, MatvecF16Fn fn) {
        f16_registry()[name] = fn;
    }

    bool set_matvec_f16_impl_by_name(const char *name) {
        const auto &r = f16_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_f16_current = it->second;
        g_f16_current_name = name;
        return true;
    }

    const char *matvec_f16_impl_name() {
        return g_f16_current_name.empty() ? "ref" : g_f16_current_name.c_str();
    }

    const char *available_matvec_f16_impls() {
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : f16_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    void matvec_f16(const uint16_t *w, const float *x, float *y, int out_dim, int in_dim) {
        MatvecF16Fn fn = g_f16_current;
        if (!fn) {
            auto it = f16_registry().find("ref");
            if (it == f16_registry().end()) {
                std::fprintf(stderr,
                             "tinyqwen: matvec_f16 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;
        }
        fn(w, x, y, out_dim, in_dim);
    }

    void register_matvec_f16_pair_impl(const char *name, MatvecPairF16Fn fn) {
        f16_pair_registry()[name] = fn;
    }

    void matvec_pair_f16(const uint16_t *w1, const uint16_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        const auto &pr = f16_pair_registry();
        auto it = pr.find(matvec_f16_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim);
            return;
        }
        matvec_f16(w1, x, y1, out_dim, in_dim);
        matvec_f16(w2, x, y2, out_dim, in_dim);
    }

    // ---- qkv 三路融合 ----
    namespace {
        std::unordered_map<std::string, MatvecQkvFn> &qkv_registry() {
            static std::unordered_map<std::string, MatvecQkvFn> r;
            return r;
        }

        std::unordered_map<std::string, MatvecQkvF16Fn> &qkv_f16_registry() {
            static std::unordered_map<std::string, MatvecQkvF16Fn> r;
            return r;
        }
    } // namespace

    void register_matvec_qkv_impl(const char *name, MatvecQkvFn fn) {
        qkv_registry()[name] = fn;
    }

    void matvec_qkv_f32(const float *wq, const float *wk, const float *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim) {
        const auto &r = qkv_registry();
        auto it = r.find(matvec_impl_name());
        if (it != r.end()) {
            it->second(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim);
            return;
        }
        matvec_f32(wq, x, yq, q_dim, in_dim);
        matvec_pair_f32(wk, wv, x, yk, yv, kv_dim, in_dim);
    }

    void register_matvec_qkv_f16_impl(const char *name, MatvecQkvF16Fn fn) {
        qkv_f16_registry()[name] = fn;
    }

    void matvec_qkv_f16(const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim) {
        const auto &r = qkv_f16_registry();
        auto it = r.find(matvec_f16_impl_name());
        if (it != r.end()) {
            it->second(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim);
            return;
        }
        matvec_f16(wq, x, yq, q_dim, in_dim);
        matvec_pair_f16(wk, wv, x, yk, yv, kv_dim, in_dim);
    }

    // ---- INT4 路径：与 f32/f16 对称的第三套注册表/选择器/入口 ----
    namespace {
        std::unordered_map<std::string, MatvecI4Fn> &i4_registry() {
            static std::unordered_map<std::string, MatvecI4Fn> r;
            return r;
        }

        std::unordered_map<std::string, MatvecPairI4Fn> &i4_pair_registry() {
            static std::unordered_map<std::string, MatvecPairI4Fn> r;
            return r;
        }

        std::unordered_map<std::string, MatvecQkvI4Fn> &qkv_i4_registry() {
            static std::unordered_map<std::string, MatvecQkvI4Fn> r;
            return r;
        }

        MatvecI4Fn g_i4_current = nullptr;
        std::string g_i4_current_name;
    } // namespace

    void register_matvec_i4_impl(const char *name, MatvecI4Fn fn) {
        i4_registry()[name] = fn;
    }

    bool set_matvec_i4_impl_by_name(const char *name) {
        const auto &r = i4_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_i4_current = it->second;
        g_i4_current_name = name;
        return true;
    }

    const char *matvec_i4_impl_name() {
        return g_i4_current_name.empty() ? "ref" : g_i4_current_name.c_str();
    }

    const char *available_matvec_i4_impls() {
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : i4_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    void matvec_i4(const uint8_t *w, const float *x, float *y,
                   int out_dim, int in_dim, int group_size) {
        MatvecI4Fn fn = g_i4_current;
        if (!fn) {
            auto it = i4_registry().find("ref");
            if (it == i4_registry().end()) {
                std::fprintf(stderr,
                             "tinyqwen: matvec_i4 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;
        }
        fn(w, x, y, out_dim, in_dim, group_size);
    }

    void register_matvec_i4_pair_impl(const char *name, MatvecPairI4Fn fn) {
        i4_pair_registry()[name] = fn;
    }

    void matvec_pair_i4(const uint8_t *w1, const uint8_t *w2, const float *x,
                        float *y1, float *y2, int out_dim, int in_dim, int group_size) {
        const auto &pr = i4_pair_registry();
        auto it = pr.find(matvec_i4_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim, group_size);
            return;
        }
        matvec_i4(w1, x, y1, out_dim, in_dim, group_size);
        matvec_i4(w2, x, y2, out_dim, in_dim, group_size);
    }

    void register_matvec_qkv_i4_impl(const char *name, MatvecQkvI4Fn fn) {
        qkv_i4_registry()[name] = fn;
    }

    void matvec_qkv_i4(const uint8_t *wq, const uint8_t *wk, const uint8_t *wv,
                       const float *x, float *yq, float *yk, float *yv,
                       int q_dim, int kv_dim, int in_dim, int group_size) {
        const auto &r = qkv_i4_registry();
        auto it = r.find(matvec_i4_impl_name());
        if (it != r.end()) {
            it->second(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim, group_size);
            return;
        }
        matvec_i4(wq, x, yq, q_dim, in_dim, group_size);
        matvec_pair_i4(wk, wv, x, yk, yv, kv_dim, in_dim, group_size);
    }

    // ---- Matmul (GEMM) 分发 ----
    namespace {
        std::unordered_map<std::string, MatmulFn> &mm_registry() {
            static std::unordered_map<std::string, MatmulFn> r;
            return r;
        }
        MatmulFn g_mm_current = nullptr;

        std::unordered_map<std::string, MatmulI4Fn> &mm_i4_registry() {
            static std::unordered_map<std::string, MatmulI4Fn> r;
            return r;
        }
        MatmulI4Fn g_mm_i4_current = nullptr;
    } // namespace

    void register_matmul_impl(const char *name, MatmulFn fn) {
        mm_registry()[name] = fn;
        if (!g_mm_current) g_mm_current = fn;
    }

    bool set_matmul_impl_by_name(const char *name) {
        auto &r = mm_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_mm_current = it->second;
        return true;
    }

    void matmul_f32(const float *w, const float *x, float *y, int M, int K, int N) {
        g_mm_current(w, x, y, M, K, N);
    }

    void register_matmul_i4_impl(const char *name, MatmulI4Fn fn) {
        mm_i4_registry()[name] = fn;
        if (!g_mm_i4_current) g_mm_i4_current = fn;
    }

    void matmul_i4(const uint8_t *w, const float *x, float *y,
                   int M, int K, int N, int group_size) {
        g_mm_i4_current(w, x, y, M, K, N, group_size);
    }

    // ---- 非 matvec 算子分发 ----
    namespace {
        std::unordered_map<std::string, RmsnormFn> &rmsnorm_registry() {
            static std::unordered_map<std::string, RmsnormFn> r;
            return r;
        }
        std::unordered_map<std::string, RopeFn> &rope_registry() {
            static std::unordered_map<std::string, RopeFn> r;
            return r;
        }
        std::unordered_map<std::string, AttentionDecodeFn> &attention_registry() {
            static std::unordered_map<std::string, AttentionDecodeFn> r;
            return r;
        }
        std::unordered_map<std::string, SwigluFn> &swiglu_registry() {
            static std::unordered_map<std::string, SwigluFn> r;
            return r;
        }
        std::unordered_map<std::string, ArgmaxFn> &argmax_registry() {
            static std::unordered_map<std::string, ArgmaxFn> r;
            return r;
        }

        // GDN 算子注册表。
        std::unordered_map<std::string, CausalConv1dUpdateFn> &conv1d_registry() {
            static std::unordered_map<std::string, CausalConv1dUpdateFn> r;
            return r;
        }
        std::unordered_map<std::string, L2normInplaceFn> &l2norm_registry() {
            static std::unordered_map<std::string, L2normInplaceFn> r;
            return r;
        }
        std::unordered_map<std::string, GdnStepFn> &gdn_step_registry() {
            static std::unordered_map<std::string, GdnStepFn> r;
            return r;
        }
        std::unordered_map<std::string, RmsnormGatedFn> &rmsnorm_gated_registry() {
            static std::unordered_map<std::string, RmsnormGatedFn> r;
            return r;
        }

        // 所有非 matvec 算子共用的当前实现名。空 = 未显式选择，通用入口兜底到 _ref。
        std::string g_ops_name;
    } // namespace

    void register_rmsnorm_impl(const char *name, RmsnormFn fn) { rmsnorm_registry()[name] = fn; }
    void register_rope_impl(const char *name, RopeFn fn) { rope_registry()[name] = fn; }
    void register_attention_decode_impl(const char *name, AttentionDecodeFn fn) {
        attention_registry()[name] = fn;
    }
    void register_swiglu_impl(const char *name, SwigluFn fn) { swiglu_registry()[name] = fn; }
    void register_argmax_impl(const char *name, ArgmaxFn fn) { argmax_registry()[name] = fn; }

    void register_causal_conv1d_update_impl(const char *name, CausalConv1dUpdateFn fn) {
        conv1d_registry()[name] = fn;
    }
    void register_l2norm_inplace_impl(const char *name, L2normInplaceFn fn) {
        l2norm_registry()[name] = fn;
    }
    void register_gdn_step_impl(const char *name, GdnStepFn fn) {
        gdn_step_registry()[name] = fn;
    }
    void register_rmsnorm_gated_impl(const char *name, RmsnormGatedFn fn) {
        rmsnorm_gated_registry()[name] = fn;
    }

    bool set_ops_impl_by_name(const char *name) {
        // "ref" = 兜底默认：清空当前名（通用入口自动落回各自 _ref），恒接受。
        if (std::string(name) == "ref") {
            g_ops_name.clear();
            return true;
        }
        // 其余名字：只要任一算子注册了该名就接受（允许"部分算子有变体、
        // 其余兜底 ref"）。
        const std::string n = name;
        const bool any = rmsnorm_registry().count(n) || rope_registry().count(n) ||
                         attention_registry().count(n) || swiglu_registry().count(n) ||
                         argmax_registry().count(n) ||
                         conv1d_registry().count(n) || l2norm_registry().count(n) ||
                         gdn_step_registry().count(n) || rmsnorm_gated_registry().count(n);
        if (!any) return false;
        g_ops_name = n;
        return true;
    }

    const char *ops_impl_name() { return g_ops_name.empty() ? "ref" : g_ops_name.c_str(); }

    void rmsnorm(const float *x, const float *weight, float *y, int n, float eps) {
        const auto &r = rmsnorm_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, weight, y, n, eps);
            return;
        }
        rmsnorm_ref(x, weight, y, n, eps); // 兜底：数值锚点
    }

    void rope(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta) {
        const auto &r = rope_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(q, k, n_heads, n_kv_heads, head_dim, pos, theta);
            return;
        }
        rope_ref(q, k, n_heads, n_kv_heads, head_dim, pos, theta);
    }

    void attention_decode(const float *q, const float *k_cache, const float *v_cache,
                          int seq_len, int max_seq_len, int n_heads, int n_kv_heads,
                          int head_dim, float scale, float *out) {
        const auto &r = attention_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(q, k_cache, v_cache, seq_len, max_seq_len, n_heads, n_kv_heads, head_dim,
                       scale, out);
            return;
        }
        attention_decode_ref(q, k_cache, v_cache, seq_len, max_seq_len, n_heads, n_kv_heads,
                             head_dim, scale, out);
    }

    void swiglu(float *gate, const float *up, int n) {
        const auto &r = swiglu_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(gate, up, n);
            return;
        }
        swiglu_ref(gate, up, n);
    }

    int argmax(const float *logits, int n) {
        const auto &r = argmax_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            return it->second(logits, n);
        }
        return argmax_ref(logits, n);
    }

    // ---- GPU decode engine 分发 ----
    namespace {
        // 一个 engine 实现的五个函数指针。
        struct GpuDecodeImpl {
            GpuDecodeCreateFn create = nullptr;
            GpuDecodeStepFn step = nullptr;
            GpuDecodeResetFn reset = nullptr;
            GpuDecodeDestroyFn destroy = nullptr;
            GpuDecodeLogitsFn logits = nullptr;
        };

        std::unordered_map<std::string, GpuDecodeImpl> &gpu_decode_registry() {
            static std::unordered_map<std::string, GpuDecodeImpl> r;
            return r;
        }

        // 当前选择。nullptr = 未选择，gpu_decode_create 返回 false（回退 CPU）。
        const GpuDecodeImpl *g_gpu_current = nullptr;
        std::string g_gpu_current_name; // 空 = 未选择
    } // namespace

    void register_gpu_decode_impl(const char *name, GpuDecodeCreateFn create,
                                  GpuDecodeStepFn step, GpuDecodeResetFn reset,
                                  GpuDecodeDestroyFn destroy, GpuDecodeLogitsFn logits) {
        gpu_decode_registry()[name] = GpuDecodeImpl{create, step, reset, destroy, logits};
    }

    bool set_gpu_decode_impl_by_name(const char *name) {
        const auto &r = gpu_decode_registry();
        auto it = r.find(name);
        if (it == r.end()) return false; // 未知名字：不改变当前选择
        g_gpu_current = &it->second;
        g_gpu_current_name = name;
        return true;
    }

    const char *gpu_decode_impl_name() { return g_gpu_current_name.c_str(); }

    const char *available_gpu_decode_impls() {
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : gpu_decode_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    bool gpu_decode_available() { return !gpu_decode_registry().empty(); }

    bool gpu_decode_create(const void *model_file, int max_seq_len, std::string *err,
                           GpuDecodeEngine **out) {
        if (!g_gpu_current || !g_gpu_current->create) return false; // 未选择 → 回退 CPU
        return g_gpu_current->create(model_file, max_seq_len, err, out);
    }

    int gpu_decode_step(GpuDecodeEngine *e, int token_id) { return g_gpu_current->step(e, token_id); }
    void gpu_decode_reset(GpuDecodeEngine *e) { g_gpu_current->reset(e); }
    void gpu_decode_destroy(GpuDecodeEngine *e) { g_gpu_current->destroy(e); }
    const float *gpu_decode_logits(const GpuDecodeEngine *e) {
        return g_gpu_current->logits(e);
    }

    // ---- GDN 算子通用入口 ----

    void causal_conv1d_update(const float *x, float *conv_state, const float *weight,
                              float *out, int dim, int kernel_size) {
        const auto &r = conv1d_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, conv_state, weight, out, dim, kernel_size);
            return;
        }
        causal_conv1d_update_ref(x, conv_state, weight, out, dim, kernel_size);
    }

    void l2norm_inplace(float *x, int n, float eps) {
        const auto &r = l2norm_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, n, eps);
            return;
        }
        l2norm_inplace_ref(x, n, eps);
    }

    void gdn_step(float *S, const float *q, const float *k, const float *v,
                  float g, float beta, float *o, int qk_dim, int v_dim) {
        const auto &r = gdn_step_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(S, q, k, v, g, beta, o, qk_dim, v_dim);
            return;
        }
        gdn_step_ref(S, q, k, v, g, beta, o, qk_dim, v_dim);
    }

    void rmsnorm_gated(const float *x, const float *gate, const float *weight,
                       float *y, int n, float eps) {
        const auto &r = rmsnorm_gated_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, gate, weight, y, n, eps);
            return;
        }
        rmsnorm_gated_ref(x, gate, weight, y, n, eps);
    }
} // namespace tinyqwen
