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
} // namespace tinyqwen
