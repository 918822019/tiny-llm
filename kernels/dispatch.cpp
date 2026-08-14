#include "dispatch.h"

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
} // namespace tinyqwen
