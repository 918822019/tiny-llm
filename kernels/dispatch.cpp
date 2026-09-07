// ============================================================================
// dispatch.cpp — 算子分发层的实现
// ============================================================================
//
// 本文件实现 tinyqwen 的算子分发（dispatch）机制。模型推理代码（model）只调用
// "通用入口"函数（如 matvec_f32、rmsnorm、gdn_step 等），由分发层根据当前配置
// 选择已注册的具体实现变体（ref / neon / neon_mt_kv_nt 等）。
//
// 核心设计：自注册 + 运行时查找
//   - 每个实现变体在自己的 .cpp 文件末尾用 TINYQWEN_MATVEC_VARIANT 等宏声明一行，
//     该宏展开为一个静态 bool 变量的初始化器，在 main() 之前执行 register_xxx_impl()。
//   - 运行时通过 set_matvec_impl_by_name("neon") 等函数选择当前实现。
//   - 通用入口按当前名字查自己的注册表；未找到则兜底到 _ref 实现。
//
// 线程安全考虑：
//   - 注册发生在 main 之前的静态初始化阶段（单线程），之后注册表只读不改。
//   - Meyers singleton（static local）保证 C++11 起线程安全的构造。
//   - g_current 等全局选择在启动时设置一次，推理期间不变。
//   - 如果未来需要运行时切换实现，需加锁或改为 atomic。
//
// 兜底策略：
//   - matvec 系列：g_current == nullptr 时自动查 "ref"；ref 也没注册则 abort。
//   - ops 系列（rmsnorm/rope/attention/swiglu/argmax/GDN 算子）：
//     按 ops_impl_name() 查注册表，未找到则直接调用对应的 _ref 函数。
//     "ref" = 清空当前名（各入口自动落回各自 _ref），恒接受。
//   - pair/qkv 融合入口：当前 impl 没注册 pair 变体就分开调两次/三次基础入口。
//
// 注册表结构：
//   每种数据类型/算子类型有独立的 unordered_map<string, FnPtr>。
//   f32/f16/i4 各有独立的选择变量和注册表，互不干扰。
//   非 matvec 算子共用一个实现名（--ops-impl neon 同时启用所有 NEON 算子）。
// ============================================================================

#include "dispatch.h"   // 分发层头文件：声明所有通用入口、注册函数、类型别名
#include "gdn_ops.h"    // GDN 算子声明：包含 *_ref 兜底函数的签名

#include <algorithm>    // std::sort（available_impls 排序输出）
#include <cstdio>       // std::fprintf（错误报告）
#include <cstdlib>      // std::abort（致命错误时终止）
#include <string>       // std::string
#include <unordered_map> // std::unordered_map（注册表容器）
#include <vector>       // std::vector（排序临时容器）

namespace tinyqwen {
    namespace {  // 匿名命名空间：内部状态不导出符号表
        // ================================================================
        // matvec f32 注册表
        // ================================================================
        // 名字 -> 实现函数指针的映射。Meyers singleton 模式：
        // static local 变量在首次调用时构造，C++11 起保证线程安全，
        // 且一定先于任何 register_matvec_impl 调用存在（避免静态初始化顺序坑）。
        std::unordered_map<std::string, MatvecFn> &registry() {
            static std::unordered_map<std::string, MatvecFn> r;  // 持久化注册表
            return r;
        }

        // 当前选择的 matvec f32 实现。nullptr = 未显式选择，matvec_f32 会兜底到 "ref"。
        MatvecFn g_current = nullptr;
        std::string g_current_name;  // 当前实现的名字。空 = 未显式选择
    } // namespace

    // ================================================================
    // register_matvec_impl — 注册一个 matvec f32 实现
    // ================================================================
    // 参数：
    //   name — 实现名字（如 "ref"、"neon"、"neon_mt_kv_nt"）
    //   fn   — 实现函数指针
    // 说明：通常不直接调用，而是用文件末尾的 TINYQWEN_MATVEC_VARIANT 宏。
    void register_matvec_impl(const char *name, MatvecFn fn) { registry()[name] = fn; }

    // ================================================================
    // set_matvec_impl_by_name — 按名字选择 matvec f32 实现
    // ================================================================
    // 参数：name — 要选择的实现名字
    // 返回值：找到返回 true 并更新当前选择；未找到返回 false 且不改变当前选择。
    // 说明：由调用方（main 或配置解析）负责报错。
    bool set_matvec_impl_by_name(const char *name) {
        const auto &r = registry();               // 引用注册表
        auto it = r.find(name);                    // 查找指定名字
        if (it == r.end()) return false;           // 未知名字：不改变当前选择，由调用方报错
        g_current = it->second;                    // 更新当前函数指针
        g_current_name = name;                     // 更新当前名字
        return true;
    }

    // ================================================================
    // matvec_impl_name — 获取当前 matvec f32 实现的名字
    // ================================================================
    // 返回值：当前实现名字；未显式选择时返回 "ref"（matvec_f32 会兜底到 ref）。
    const char *matvec_impl_name() {
        return g_current_name.empty() ? "ref" : g_current_name.c_str();
    }

    // ================================================================
    // available_matvec_impls — 获取所有已注册的 matvec f32 实现名
    // ================================================================
    // 返回值：逗号分隔的实现名列表（已排序，保证输出稳定）。
    // 说明：注册都发生在 main 之前且之后不变，拼一次缓存起来。
    const char *available_matvec_impls() {
        static std::string joined;  // 缓存拼接结果
        if (joined.empty()) {       // 首次调用时构建
            std::vector<std::string> names;          // 收集所有名字
            for (const auto &kv : registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());   // 排序保证稳定输出
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";               // 非首个元素前加逗号
                joined += names[i];                   // 追加名字
            }
        }
        return joined.c_str();
    }

    // ================================================================
    // matvec_f32 — matvec f32 的通用入口
    // ================================================================
    // 功能：model 调用此函数而非直接调某个具体实现。
    // 逻辑：优先使用 g_current；若为 nullptr 则兜底查 "ref"；ref 都没注册则 abort。
    // 参数：与 MatvecFn 签名一致（w, x, y, out_dim, in_dim）。
    void matvec_f32(const float *w, const float *x, float *y, int out_dim, int in_dim) {
        MatvecFn fn = g_current;   // 读取当前选择
        if (!fn) {
            // 兜底到 ref：只要 ref 实现被链接进来，行为就和 v1 完全一致。
            auto it = registry().find("ref");
            if (it == registry().end()) {
                // ref 都没注册 → 链接配置错误，无法继续
                std::fprintf(stderr,
                             "tinyqwen: matvec 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;  // 使用 ref 实现
        }
        fn(w, x, y, out_dim, in_dim);  // 调用选中的实现
    }

    namespace {
        // ================================================================
        // matvec pair f32 注册表
        // ================================================================
        // pair 实现注册表：key 与 matvec 实现同名。注册同样发生在 main 之前。
        // pair 入口将 k_proj/v_proj 合并为一次调用，有机会摊薄同步开销。
        std::unordered_map<std::string, MatvecPairFn> &pair_registry() {
            static std::unordered_map<std::string, MatvecPairFn> r;
            return r;
        }
    } // namespace

    // ================================================================
    // register_matvec_pair_impl — 注册一个 matvec pair f32 实现
    // ================================================================
    void register_matvec_pair_impl(const char *name, MatvecPairFn fn) {
        pair_registry()[name] = fn;  // 以同名 key 登记进 pair 注册表
    }

    // ================================================================
    // matvec_pair_f32 — matvec pair f32 的通用入口
    // ================================================================
    // 功能：y1 = W1 @ x, y2 = W2 @ x（同一个输入 x，两个权重矩阵）。
    // 逻辑：当前 impl 注册了 pair 实现就用它；没注册则兜底为调两次 matvec_f32。
    //        兜底路径数值与分开调用完全一致，不关心此优化的 impl 无需注册任何东西。
    void matvec_pair_f32(const float *w1, const float *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        // 按当前 impl 名查 pair 注册表
        const auto &pr = pair_registry();
        auto it = pr.find(matvec_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim);  // 使用 pair 实现
            return;
        }
        // 兜底：等价于调用方分开调两次 matvec_f32（数值一致，行为不变）。
        matvec_f32(w1, x, y1, out_dim, in_dim);  // y1 = W1 @ x
        matvec_f32(w2, x, y2, out_dim, in_dim);  // y2 = W2 @ x
    }

    // ====================================================================
    // f16 路径：与 f32 完全对称的第二套注册表/选择器/入口
    // ====================================================================
    // f16 实现有独立注册表，实现名与 f32 共享同一命名空间。
    // 两表独立保证 "f32 模型配 f16 实现名" 会 fail fast，而不是静默兜底。
    namespace {
        // f16 matvec 注册表
        std::unordered_map<std::string, MatvecF16Fn> &f16_registry() {
            static std::unordered_map<std::string, MatvecF16Fn> r;
            return r;
        }

        // f16 pair matvec 注册表
        std::unordered_map<std::string, MatvecPairF16Fn> &f16_pair_registry() {
            static std::unordered_map<std::string, MatvecPairF16Fn> r;
            return r;
        }

        // f16 当前选择
        MatvecF16Fn g_f16_current = nullptr;       // nullptr = 未显式选择
        std::string g_f16_current_name;             // 空 = 未显式选择
    } // namespace

    // 注册 f16 matvec 实现
    void register_matvec_f16_impl(const char *name, MatvecF16Fn fn) {
        f16_registry()[name] = fn;
    }

    // 按名字选择 f16 matvec 实现
    bool set_matvec_f16_impl_by_name(const char *name) {
        const auto &r = f16_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;           // 未知名字不改变当前选择
        g_f16_current = it->second;                 // 更新函数指针
        g_f16_current_name = name;                  // 更新名字
        return true;
    }

    // 获取当前 f16 matvec 实现名
    const char *matvec_f16_impl_name() {
        return g_f16_current_name.empty() ? "ref" : g_f16_current_name.c_str();
    }

    // 获取所有已注册的 f16 matvec 实现名
    const char *available_matvec_f16_impls() {
        static std::string joined;  // 缓存
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : f16_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());   // 排序保证稳定
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    // f16 matvec 通用入口：未显式选择时兜底到 f16 注册表里的 "ref"
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

    // 注册 f16 pair matvec 实现
    void register_matvec_f16_pair_impl(const char *name, MatvecPairF16Fn fn) {
        f16_pair_registry()[name] = fn;
    }

    // f16 pair matvec 通用入口：未注册 pair 的 impl 兜底为调两次 matvec_f16
    void matvec_pair_f16(const uint16_t *w1, const uint16_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        const auto &pr = f16_pair_registry();
        auto it = pr.find(matvec_f16_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim);  // 使用 pair 实现
            return;
        }
        // 兜底：分开调两次
        matvec_f16(w1, x, y1, out_dim, in_dim);
        matvec_f16(w2, x, y2, out_dim, in_dim);
    }

    // ====================================================================
    // qkv 三路融合：q + k + v 共享输入向量，一次 fork-join
    // ====================================================================
    // q_dim 和 kv_dim 可以不同（Qwen: 896 vs 128）。
    // 兜底：matvec(q) + matvec_pair(k,v)。
    namespace {
        // qkv f32 注册表
        std::unordered_map<std::string, MatvecQkvFn> &qkv_registry() {
            static std::unordered_map<std::string, MatvecQkvFn> r;
            return r;
        }

        // qkv f16 注册表
        std::unordered_map<std::string, MatvecQkvF16Fn> &qkv_f16_registry() {
            static std::unordered_map<std::string, MatvecQkvF16Fn> r;
            return r;
        }
    } // namespace

    // 注册 qkv f32 实现
    void register_matvec_qkv_impl(const char *name, MatvecQkvFn fn) {
        qkv_registry()[name] = fn;
    }

    // qkv f32 通用入口：未注册则兜底 matvec(q) + matvec_pair(k,v)
    void matvec_qkv_f32(const float *wq, const float *wk, const float *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim) {
        const auto &r = qkv_registry();
        auto it = r.find(matvec_impl_name());
        if (it != r.end()) {
            it->second(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim);
            return;
        }
        // 兜底：q 单独调 + k,v 用 pair
        matvec_f32(wq, x, yq, q_dim, in_dim);
        matvec_pair_f32(wk, wv, x, yk, yv, kv_dim, in_dim);
    }

    // 注册 qkv f16 实现
    void register_matvec_qkv_f16_impl(const char *name, MatvecQkvF16Fn fn) {
        qkv_f16_registry()[name] = fn;
    }

    // qkv f16 通用入口：未注册则兜底 matvec_f16(q) + matvec_pair_f16(k,v)
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

    // ====================================================================
    // INT4 路径：与 f32/f16 对称的第三套注册表/选择器/入口
    // ====================================================================
    // INT4 weight-only 量化：权重 uint4 packed，per-group scale+zero，
    // interleaved 布局。比 f32/f16 多一个 group_size 参数。
    namespace {
        // INT4 matvec 注册表
        std::unordered_map<std::string, MatvecI4Fn> &i4_registry() {
            static std::unordered_map<std::string, MatvecI4Fn> r;
            return r;
        }

        // INT4 pair matvec 注册表
        std::unordered_map<std::string, MatvecPairI4Fn> &i4_pair_registry() {
            static std::unordered_map<std::string, MatvecPairI4Fn> r;
            return r;
        }

        // INT4 qkv 注册表
        std::unordered_map<std::string, MatvecQkvI4Fn> &qkv_i4_registry() {
            static std::unordered_map<std::string, MatvecQkvI4Fn> r;
            return r;
        }

        // INT4 当前选择
        MatvecI4Fn g_i4_current = nullptr;       // nullptr = 未显式选择
        std::string g_i4_current_name;            // 空 = 未显式选择
    } // namespace

    // 注册 INT4 matvec 实现
    void register_matvec_i4_impl(const char *name, MatvecI4Fn fn) {
        i4_registry()[name] = fn;
    }

    // 按名字选择 INT4 matvec 实现
    bool set_matvec_i4_impl_by_name(const char *name) {
        const auto &r = i4_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_i4_current = it->second;
        g_i4_current_name = name;
        return true;
    }

    // 获取当前 INT4 matvec 实现名
    const char *matvec_i4_impl_name() {
        return g_i4_current_name.empty() ? "ref" : g_i4_current_name.c_str();
    }

    // 获取所有已注册的 INT4 matvec 实现名
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

    // INT4 matvec 通用入口：未显式选择时兜底到 "ref"
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

    // 注册 INT4 pair matvec 实现
    void register_matvec_i4_pair_impl(const char *name, MatvecPairI4Fn fn) {
        i4_pair_registry()[name] = fn;
    }

    // INT4 pair matvec 通用入口
    void matvec_pair_i4(const uint8_t *w1, const uint8_t *w2, const float *x,
                        float *y1, float *y2, int out_dim, int in_dim, int group_size) {
        const auto &pr = i4_pair_registry();
        auto it = pr.find(matvec_i4_impl_name());
        if (it != pr.end()) {
            it->second(w1, w2, x, y1, y2, out_dim, in_dim, group_size);
            return;
        }
        // 兜底：分开调两次
        matvec_i4(w1, x, y1, out_dim, in_dim, group_size);
        matvec_i4(w2, x, y2, out_dim, in_dim, group_size);
    }

    // 注册 INT4 qkv 实现
    void register_matvec_qkv_i4_impl(const char *name, MatvecQkvI4Fn fn) {
        qkv_i4_registry()[name] = fn;
    }

    // INT4 qkv 通用入口
    void matvec_qkv_i4(const uint8_t *wq, const uint8_t *wk, const uint8_t *wv,
                       const float *x, float *yq, float *yk, float *yv,
                       int q_dim, int kv_dim, int in_dim, int group_size) {
        const auto &r = qkv_i4_registry();
        auto it = r.find(matvec_i4_impl_name());
        if (it != r.end()) {
            it->second(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim, group_size);
            return;
        }
        // 兜底：q 单独 + k,v pair
        matvec_i4(wq, x, yq, q_dim, in_dim, group_size);
        matvec_pair_i4(wk, wv, x, yk, yv, kv_dim, in_dim, group_size);
    }

    // ====================================================================
    // VQ2 路径：与 i4 对称的第四套注册表/选择器/入口
    // ====================================================================
    // 2-bit 向量量化：权重 = [码本 256×fp16][uint8 索引]。反量化纯查表。
    // pair/qkv/matmul 无融合内核，通用入口直接拆成多次 matvec_vq2。
    namespace {
        // VQ2 matvec 注册表
        std::unordered_map<std::string, MatvecVQ2Fn> &vq2_registry() {
            static std::unordered_map<std::string, MatvecVQ2Fn> r;
            return r;
        }
        // VQ2 当前选择
        MatvecVQ2Fn g_vq2_current = nullptr;     // nullptr = 未显式选择
        std::string g_vq2_current_name;           // 空 = 未显式选择
    } // namespace

    // 注册 VQ2 matvec 实现
    void register_matvec_vq2_impl(const char *name, MatvecVQ2Fn fn) {
        vq2_registry()[name] = fn;
    }

    // 按名字选择 VQ2 matvec 实现
    bool set_matvec_vq2_impl_by_name(const char *name) {
        const auto &r = vq2_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_vq2_current = it->second;
        g_vq2_current_name = name;
        return true;
    }

    // 获取当前 VQ2 matvec 实现名
    const char *matvec_vq2_impl_name() {
        return g_vq2_current_name.empty() ? "ref" : g_vq2_current_name.c_str();
    }

    // 获取所有已注册的 VQ2 matvec 实现名
    const char *available_matvec_vq2_impls() {
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : vq2_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    // VQ2 matvec 通用入口：未显式选择时兜底到 "ref"
    void matvec_vq2(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim) {
        MatvecVQ2Fn fn = g_vq2_current;
        if (!fn) {
            auto it = vq2_registry().find("ref");
            if (it == vq2_registry().end()) {
                std::fprintf(stderr,
                             "tinyqwen: matvec_vq2 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;
        }
        fn(w, x, y, out_dim, in_dim);
    }

    // VQ2 pair 通用入口：无融合内核，拆成两次 matvec_vq2
    void matvec_pair_vq2(const uint8_t *w1, const uint8_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        matvec_vq2(w1, x, y1, out_dim, in_dim);
        matvec_vq2(w2, x, y2, out_dim, in_dim);
    }

    // VQ2 qkv 通用入口：无融合内核，拆成三次 matvec_vq2
    void matvec_qkv_vq2(const uint8_t *wq, const uint8_t *wk, const uint8_t *wv,
                        const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim) {
        matvec_vq2(wq, x, yq, q_dim, in_dim);
        matvec_vq2(wk, x, yk, kv_dim, in_dim);
        matvec_vq2(wv, x, yv, kv_dim, in_dim);
    }

    // VQ2 matmul 通用入口：X/Y 列主序（每列一个 token），逐列调 matvec_vq2
    void matmul_vq2(const uint8_t *w, const float *x, float *y, int M, int K, int N) {
        for (int n = 0; n < N; ++n) {
            matvec_vq2(w, x + static_cast<size_t>(n) * K,
                       y + static_cast<size_t>(n) * M, M, K);
        }
    }

    // ====================================================================
    // GPTQ-INT4 路径：与 i4 对称的第五套注册表/选择器/入口
    // ====================================================================
    // 原生 AutoGPTQ 打包：in-band 存 scales/qzeros/(g_idx)/qweight（见
    // tiny_format.h 的 GptqBlockOffsets）。pair/qkv 无融合内核，拆成多次
    // matvec_gptq。签名带 group_size（与 i4 同）。
    namespace {
        // GPTQ matvec 注册表
        std::unordered_map<std::string, MatvecGPTQFn> &gptq_registry() {
            static std::unordered_map<std::string, MatvecGPTQFn> r;
            return r;
        }
        // GPTQ 当前选择
        MatvecGPTQFn g_gptq_current = nullptr;   // nullptr = 未显式选择
        std::string g_gptq_current_name;          // 空 = 未显式选择
    } // namespace

    // 注册 GPTQ matvec 实现
    void register_matvec_gptq_impl(const char *name, MatvecGPTQFn fn) {
        gptq_registry()[name] = fn;
    }

    // 按名字选择 GPTQ matvec 实现
    bool set_matvec_gptq_impl_by_name(const char *name) {
        const auto &r = gptq_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_gptq_current = it->second;
        g_gptq_current_name = name;
        return true;
    }

    // 获取当前 GPTQ matvec 实现名
    const char *matvec_gptq_impl_name() {
        return g_gptq_current_name.empty() ? "ref" : g_gptq_current_name.c_str();
    }

    // 获取所有已注册的 GPTQ matvec 实现名
    const char *available_matvec_gptq_impls() {
        static std::string joined;
        if (joined.empty()) {
            std::vector<std::string> names;
            for (const auto &kv : gptq_registry()) names.push_back(kv.first);
            std::sort(names.begin(), names.end());
            for (size_t i = 0; i < names.size(); ++i) {
                if (i) joined += ", ";
                joined += names[i];
            }
        }
        return joined.c_str();
    }

    // GPTQ matvec 通用入口：未显式选择时兜底到 "ref"
    void matvec_gptq(const uint8_t *w, const float *x, float *y,
                     int out_dim, int in_dim, int group_size) {
        MatvecGPTQFn fn = g_gptq_current;
        if (!fn) {
            auto it = gptq_registry().find("ref");
            if (it == gptq_registry().end()) {
                std::fprintf(stderr,
                             "tinyqwen: matvec_gptq 'ref' 未注册——检查 kernels 是否被整体链接\n");
                std::abort();
            }
            fn = it->second;
        }
        fn(w, x, y, out_dim, in_dim, group_size);
    }

    // ====================================================================
    // Matmul (GEMM) 分发：prefill 批量投影
    // ====================================================================
    // Y[M,N] = W[M,K] × X[K,N]。N=1 时退化为 matvec。
    namespace {
        // f32 matmul 注册表
        std::unordered_map<std::string, MatmulFn> &mm_registry() {
            static std::unordered_map<std::string, MatmulFn> r;
            return r;
        }
        MatmulFn g_mm_current = nullptr;  // 当前 matmul 选择

        // INT4 matmul 注册表
        std::unordered_map<std::string, MatmulI4Fn> &mm_i4_registry() {
            static std::unordered_map<std::string, MatmulI4Fn> r;
            return r;
        }
        MatmulI4Fn g_mm_i4_current = nullptr;  // 当前 INT4 matmul 选择
    } // namespace

    // 注册 matmul f32 实现；首次注册自动设为默认
    void register_matmul_impl(const char *name, MatmulFn fn) {
        mm_registry()[name] = fn;
        if (!g_mm_current) g_mm_current = fn;  // 首个注册的成为默认
    }

    // 按名字选择 matmul f32 实现
    bool set_matmul_impl_by_name(const char *name) {
        auto &r = mm_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;
        g_mm_current = it->second;
        return true;
    }

    // matmul f32 通用入口
    void matmul_f32(const float *w, const float *x, float *y, int M, int K, int N) {
        g_mm_current(w, x, y, M, K, N);  // 直接调用（必须有至少一个注册）
    }

    // 注册 INT4 matmul 实现
    void register_matmul_i4_impl(const char *name, MatmulI4Fn fn) {
        mm_i4_registry()[name] = fn;
        if (!g_mm_i4_current) g_mm_i4_current = fn;  // 首个注册的成为默认
    }

    // INT4 matmul 通用入口
    void matmul_i4(const uint8_t *w, const float *x, float *y,
                   int M, int K, int N, int group_size) {
        g_mm_i4_current(w, x, y, M, K, N, group_size);
    }

    // ====================================================================
    // 非 matvec 算子分发（ops dispatch）
    // ====================================================================
    // rmsnorm / rope / attention_decode / swiglu / argmax + GDN 四个算子
    // 各建一个小注册表，共享同一个"当前实现名"（默认 "ref"）。
    // 通用入口按当前名查自己的注册表，未注册就兜底直调对应 _ref。
    namespace {
        // ---- 传统 Transformer 算子注册表 ----
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
        std::unordered_map<std::string, BiipRotateFn> &biip_rotate_registry() {
            static std::unordered_map<std::string, BiipRotateFn> r;
            return r;
        }

        // ---- GDN 算子注册表 ----
        // Qwen3.5 的 GDN 层专属算子：causal conv1d / l2norm / gdn_step / rmsnorm_gated
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

        // MoE 路由门 top-k + softmax 注册表
        std::unordered_map<std::string, TopKSoftmaxFn> &topk_softmax_registry() {
            static std::unordered_map<std::string, TopKSoftmaxFn> r;
            return r;
        }

        // 所有非 matvec 算子共用的当前实现名。
        // 空 = 未显式选择，通用入口兜底到各自的 _ref。
        std::string g_ops_name;
    } // namespace

    // ---- 注册函数（每个一行，登记进对应注册表）----
    void register_rmsnorm_impl(const char *name, RmsnormFn fn) { rmsnorm_registry()[name] = fn; }
    void register_rope_impl(const char *name, RopeFn fn) { rope_registry()[name] = fn; }
    void register_attention_decode_impl(const char *name, AttentionDecodeFn fn) {
        attention_registry()[name] = fn;
    }
    void register_swiglu_impl(const char *name, SwigluFn fn) { swiglu_registry()[name] = fn; }
    void register_argmax_impl(const char *name, ArgmaxFn fn) { argmax_registry()[name] = fn; }
    void register_biip_rotate_impl(const char *name, BiipRotateFn fn) {
        biip_rotate_registry()[name] = fn;
    }

    // GDN 算子注册函数
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
    void register_topk_softmax_impl(const char *name, TopKSoftmaxFn fn) {
        topk_softmax_registry()[name] = fn;
    }

    // ================================================================
    // set_ops_impl_by_name — 按名字选择 ops 实现
    // ================================================================
    // 特殊规则："ref" = 清空当前名（通用入口自动落回各自 _ref），恒接受。
    // 其余名字：只要任一算子注册了该名就接受（允许"部分算子有变体、其余兜底 ref"）。
    bool set_ops_impl_by_name(const char *name) {
        // "ref" = 兜底默认：清空当前名，恒接受
        if (std::string(name) == "ref") {
            g_ops_name.clear();
            return true;
        }
        // 检查是否有任何算子注册了该名字
        const std::string n = name;
        const bool any = rmsnorm_registry().count(n) || rope_registry().count(n) ||
                         attention_registry().count(n) || swiglu_registry().count(n) ||
                         argmax_registry().count(n) || biip_rotate_registry().count(n) ||
                         conv1d_registry().count(n) || l2norm_registry().count(n) ||
                         gdn_step_registry().count(n) || rmsnorm_gated_registry().count(n) ||
                         topk_softmax_registry().count(n);
        if (!any) return false;  // 没有任何算子注册该名 → 拒绝
        g_ops_name = n;          // 接受并记录
        return true;
    }

    // 获取当前 ops 实现名（未选择时为 "ref"）
    const char *ops_impl_name() { return g_ops_name.empty() ? "ref" : g_ops_name.c_str(); }

    // ================================================================
    // 传统 Transformer 算子的通用入口
    // ================================================================
    // 每个入口的逻辑相同：按 ops_impl_name() 查注册表，找到就调用，否则兜底 _ref。

    // RMSNorm 通用入口
    void rmsnorm(const float *x, const float *weight, float *y, int n, float eps) {
        const auto &r = rmsnorm_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, weight, y, n, eps);
            return;
        }
        rmsnorm_ref(x, weight, y, n, eps);  // 兜底：数值锚点
    }

    // RoPE 通用入口
    void rope(float *q, float *k, int n_heads, int n_kv_heads, int head_dim, int pos,
              float theta) {
        const auto &r = rope_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(q, k, n_heads, n_kv_heads, head_dim, pos, theta);
            return;
        }
        rope_ref(q, k, n_heads, n_kv_heads, head_dim, pos, theta);  // 兜底
    }

    // Attention decode 通用入口
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
                             head_dim, scale, out);  // 兜底
    }

    // SwiGLU 通用入口
    void swiglu(float *gate, const float *up, int n) {
        const auto &r = swiglu_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(gate, up, n);
            return;
        }
        swiglu_ref(gate, up, n);  // 兜底
    }

    // Argmax 通用入口
    int argmax(const float *logits, int n) {
        const auto &r = argmax_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            return it->second(logits, n);
        }
        return argmax_ref(logits, n);  // 兜底
    }

    // MoE 路由门 top-k + softmax 通用入口
    void topk_softmax(const float *gate_logits, int n, int k, int *indices,
                      float *weights) {
        const auto &r = topk_softmax_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(gate_logits, n, k, indices, weights);
            return;
        }
        topk_softmax_ref(gate_logits, n, k, indices, weights);  // 兜底
    }

    // BiIP 激活旋转通用入口
    void biip_rotate_activation(const float *x, float *y, int dim,
                                const float *scale, const float *sign, int block_size) {
        const auto &r = biip_rotate_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, y, dim, scale, sign, block_size);
            return;
        }
        biip_rotate_activation_ref(x, y, dim, scale, sign, block_size);  // 兜底
    }

    // ====================================================================
    // GPU decode engine 分发
    // ====================================================================
    // 与上面"逐算子"分发不同：这是一个"整段 forward"级的可插拔入口。
    // engine 把权重/激活/KV cache 全部常驻显存，decode_step 在单条 CUDA stream
    // 上串起整个前向，只有 token id 过 PCIe——消灭逐 matvec 的 CPU↔GPU 桥接。
    namespace {
        // 一个 engine 实现的五个函数指针打包成结构体
        struct GpuDecodeImpl {
            GpuDecodeCreateFn create = nullptr;    // 创建 engine 实例
            GpuDecodeStepFn step = nullptr;         // 单步 decode
            GpuDecodeResetFn reset = nullptr;       // 重置序列状态
            GpuDecodeDestroyFn destroy = nullptr;   // 销毁 engine 实例
            GpuDecodeLogitsFn logits = nullptr;     // 获取 logits 指针
        };

        // GPU decode 注册表
        std::unordered_map<std::string, GpuDecodeImpl> &gpu_decode_registry() {
            static std::unordered_map<std::string, GpuDecodeImpl> r;
            return r;
        }

        // 当前选择。nullptr = 未选择，gpu_decode_create 返回 false（回退 CPU）。
        const GpuDecodeImpl *g_gpu_current = nullptr;
        std::string g_gpu_current_name;  // 空 = 未选择
    } // namespace

    // 注册 GPU decode engine 实现（五个函数指针一起登记）
    void register_gpu_decode_impl(const char *name, GpuDecodeCreateFn create,
                                  GpuDecodeStepFn step, GpuDecodeResetFn reset,
                                  GpuDecodeDestroyFn destroy, GpuDecodeLogitsFn logits) {
        gpu_decode_registry()[name] = GpuDecodeImpl{create, step, reset, destroy, logits};
    }

    // 按名字选择 GPU decode 实现
    bool set_gpu_decode_impl_by_name(const char *name) {
        const auto &r = gpu_decode_registry();
        auto it = r.find(name);
        if (it == r.end()) return false;  // 未知名字：不改变当前选择
        g_gpu_current = &it->second;       // 指向注册表内的结构体（生命周期由 map 保证）
        g_gpu_current_name = name;
        return true;
    }

    // 获取当前 GPU decode 实现名（未选择时为 ""）
    const char *gpu_decode_impl_name() { return g_gpu_current_name.c_str(); }

    // 获取所有已注册的 GPU decode 实现名
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

    // 是否有 GPU decode 实现可用
    bool gpu_decode_available() { return !gpu_decode_registry().empty(); }

    // GPU decode 通用入口：create 在未选择实现时返回 false（调用方回退 CPU forward）
    bool gpu_decode_create(const void *model_file, int max_seq_len, std::string *err,
                           GpuDecodeEngine **out) {
        if (!g_gpu_current || !g_gpu_current->create) return false;  // 未选择 → 回退 CPU
        return g_gpu_current->create(model_file, max_seq_len, err, out);
    }

    // GPU decode 单步
    int gpu_decode_step(GpuDecodeEngine *e, int token_id) { return g_gpu_current->step(e, token_id); }
    // GPU decode 重置
    void gpu_decode_reset(GpuDecodeEngine *e) { g_gpu_current->reset(e); }
    // GPU decode 销毁
    void gpu_decode_destroy(GpuDecodeEngine *e) { g_gpu_current->destroy(e); }
    // GPU decode 获取 logits（device 指针）
    const float *gpu_decode_logits(const GpuDecodeEngine *e) {
        return g_gpu_current->logits(e);
    }

    // ====================================================================
    // GDN 算子通用入口
    // ====================================================================
    // Qwen3.5 的 GDN 层专属算子：causal conv1d / l2norm / gdn_step / rmsnorm_gated。
    // 与传统 ops 共用同一个实现名（--ops-impl neon 同时启用所有 NEON 算子）。
    // 兜底策略相同：按 ops_impl_name() 查注册表，未找到则直调 _ref。

    // Causal Conv1D Update 通用入口
    void causal_conv1d_update(const float *x, float *conv_state, const float *weight,
                              float *out, int dim, int kernel_size) {
        const auto &r = conv1d_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, conv_state, weight, out, dim, kernel_size);
            return;
        }
        causal_conv1d_update_ref(x, conv_state, weight, out, dim, kernel_size);  // 兜底
    }

    // L2 Norm Inplace 通用入口
    void l2norm_inplace(float *x, int n, float eps) {
        const auto &r = l2norm_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, n, eps);
            return;
        }
        l2norm_inplace_ref(x, n, eps);  // 兜底
    }

    // GDN Step 通用入口
    void gdn_step(float *S, const float *q, const float *k, const float *v,
                  float g, float beta, float *o, int qk_dim, int v_dim) {
        const auto &r = gdn_step_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(S, q, k, v, g, beta, o, qk_dim, v_dim);
            return;
        }
        gdn_step_ref(S, q, k, v, g, beta, o, qk_dim, v_dim);  // 兜底
    }

    // Gated RMSNorm 通用入口
    void rmsnorm_gated(const float *x, const float *gate, const float *weight,
                       float *y, int n, float eps) {
        const auto &r = rmsnorm_gated_registry();
        auto it = r.find(ops_impl_name());
        if (it != r.end()) {
            it->second(x, gate, weight, y, n, eps);
            return;
        }
        rmsnorm_gated_ref(x, gate, weight, y, n, eps);  // 兜底
    }
} // namespace tinyqwen
