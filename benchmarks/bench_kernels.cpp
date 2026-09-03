// ============================================================================
// bench_kernels.cpp — 算子级微基准（kernel micro-benchmark）
// ============================================================================
//
// 目的：给"优化算子"提供标准评测管线。任何新变体只要放进对应算子文件夹、
// 用自注册宏登记，就能在不改本文件的情况下被枚举、校验、测速：
//
//   1. 枚举 —— 从 dispatch 各注册表读出所有已注册实现名；
//   2. 校验 —— 每个实现与 "ref"（参考实现，double 累加锚）在同一份
//      量化权重上比对，报告最大相对误差并分级（PASS / TOL / FAIL）；
//   3. 测速 —— 自适应迭代次数计时，报告 ms/call、GB/s、GFLOP/s、
//      相对 ref 的加速比；
//   4. 落盘 —— 可选 --csv / --json，供优化前后回归对比。
//
// 覆盖的 matvec 族（与 dispatch 的四套注册表一一对应）：
//   vq2（2-bit 块向量量化）/ i4（INT4 per-group）/ f16 / f32。
// 另附 BiIP 激活旋转的带宽测量（VQ2 旋转推理路径的运行时开销）。
//
// 用法示例：
//   ./bench_kernels                                  # 全族 × Qwen2.5-0.5B 形状
//   ./bench_kernels --family vq2                     # 只测 vq2
//   ./bench_kernels --family i4 --out 4864 --in 896  # 单一自定义形状
//   ./bench_kernels --list                           # 只打印已注册实现
//   ./bench_kernels --csv out.csv --budget-ms 500    # 落盘 + 加长计时预算
//
// 线程类变体（*_mt）受环境变量 TINYQWEN_MT_THREADS 控制（见各 mt 内核）。
// ============================================================================

#include "dispatch.h"   // dispatch 接口 + 自注册宏声明（内含 ref_ops.h）

#include <algorithm>    // std::max / std::min / std::sort / std::rotate
#include <chrono>       // steady_clock 计时
#include <cmath>        // std::sqrt / std::fabs / std::round
#include <cstdint>      // uint8_t / uint16_t / uint64_t
#include <cstdio>       // std::printf
#include <cstdlib>      // std::exit / std::getenv
#include <cstring>      // std::memcpy / std::strcmp
#include <fstream>      // std::ofstream（CSV 落盘）
#include <string>       // std::string
#include <vector>       // std::vector

using namespace tinyqwen;
using Clock = std::chrono::steady_clock;

// ============================================================================
// 防优化吞噬的累加槽：每次内核调用后掺一点输出，循环才不会被编译器删掉
// ============================================================================
static volatile float g_sink = 0.0f;
static void soak(const float *y, int n) { g_sink += y[0] + y[n - 1]; }

// ============================================================================
// 确定性伪随机源（LCG）：跨运行可复现，保证各实现拿到完全相同的输入
// ============================================================================
struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed) {}
    uint32_t next_u32() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    }
    float next_f32() {  // [-1, 1)
        return static_cast<float>(next_u32()) / static_cast<float>(1u << 31) - 1.0f;
    }
};

// ============================================================================
// 形状定义：name + [out_dim, in_dim]。big = 大张量（f32 族默认跳过，见 --big）
// ============================================================================
struct Shape {
    const char *name;
    int out_dim;
    int in_dim;
    bool big;
};

// Qwen2.5-0.5B 的真实投影形状（hidden=896, intermediate=4864,
// kv_dim=2kv_heads×64=128, vocab=151936）。
static const std::vector<Shape> kQwen05bShapes = {
    {"q_proj", 896, 896, false},
    {"k_proj", 128, 896, false},
    {"v_proj", 128, 896, false},
    {"o_proj", 896, 896, false},
    {"gate_up", 4864, 896, false},
    {"down", 896, 4864, false},
    {"lm_head", 151936, 896, true},  // big：f32 下 ~545MB，默认不测
};

// Qwen3-0.6B 的真实投影形状（hidden=1024, intermediate=3072,
// q_dim=16×128=2048, kv_dim=8×128=1024, vocab=151936）。旋转 VQ2 模型跑这组。
// 末两个是**融合等价形状**：把 q/k/v 与 gate/up 各自的输出维拼起来。
// 用 time(qkv_fused) vs time(q)+time(k)+time(v) 就能量出逐调用开销
// （fork-join + 线程池唤醒），回答"融合值不值得做"。
static const std::vector<Shape> kQwen3_06bShapes = {
    {"q_proj", 2048, 1024, false},
    {"k_proj", 1024, 1024, false},
    {"v_proj", 1024, 1024, false},
    {"o_proj", 1024, 2048, false},
    {"gate", 3072, 1024, false},
    {"up", 3072, 1024, false},
    {"down", 1024, 3072, false},
    {"qkv_fused", 4096, 1024, false},      // = q(2048) + k(1024) + v(1024)
    {"gate_up_fused", 6144, 1024, false},  // = gate(3072) + up(3072)
    {"lm_head", 151936, 1024, true},       // big：f32 下 ~622MB，默认不测
};

// ============================================================================
// 权重缓冲构造：同一份 f32 基础矩阵 → 各族布局，保证族内各实现输入一致
// ============================================================================

// f32 基础矩阵：均匀分布 [-1, 1)
static std::vector<float> gen_matrix(size_t n, Rng &rng) {
    std::vector<float> m(n);
    for (size_t i = 0; i < n; ++i) m[i] = rng.next_f32();
    return m;
}

// f16 布局：逐元素 float_to_half（round-to-nearest-even，与导出器一致）
static std::vector<uint16_t> to_f16(const std::vector<float> &m) {
    std::vector<uint16_t> h(m.size());
    for (size_t i = 0; i < m.size(); ++i) h[i] = float_to_half(m[i]);
    return h;
}

// INT4 interleaved 布局（与 matvec_i4_ref 完全对齐）：
// 每行 = groups_per_row × [scale_fp16(2B) | zero_fp16(2B) | packed(gs/2 B)]，
// 低 nibble 在前；反量化 = (q - zero) × scale。
static std::vector<uint8_t> pack_i4(const std::vector<float> &m,
                                    int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_bytes = 4 + group_size / 2;
    const size_t row_bytes = static_cast<size_t>(groups_per_row) * group_bytes;
    std::vector<uint8_t> buf(static_cast<size_t>(out_dim) * row_bytes, 0);
    for (int o = 0; o < out_dim; ++o) {
        const float *src = m.data() + static_cast<size_t>(o) * in_dim;
        uint8_t *row = buf.data() + static_cast<size_t>(o) * row_bytes;
        for (int g = 0; g < groups_per_row; ++g) {
            const int col = g * group_size;
            const int n_elems = std::min(group_size, in_dim - col);
            float mn = src[col], mx = src[col];
            for (int i = 1; i < n_elems; ++i) {
                mn = std::min(mn, src[col + i]);
                mx = std::max(mx, src[col + i]);
            }
            const float scale = (mx > mn) ? (mx - mn) / 15.0f : 1.0f;
            int zero = static_cast<int>(std::lround(-mn / scale));
            zero = std::max(0, std::min(15, zero));
            uint8_t *grp = row + static_cast<size_t>(g) * group_bytes;
            const uint16_t scale_h = float_to_half(scale);
            const uint16_t zero_h = float_to_half(static_cast<float>(zero));
            std::memcpy(grp, &scale_h, 2);
            std::memcpy(grp + 2, &zero_h, 2);
            for (int i = 0; i < n_elems; ++i) {
                int q = static_cast<int>(std::lround(src[col + i] / scale)) + zero;
                q = std::max(0, std::min(15, q));
                uint8_t *byte = grp + 4 + i / 2;
                if (i % 2 == 0) *byte |= static_cast<uint8_t>(q);        // 低 nibble
                else *byte |= static_cast<uint8_t>(q << 4);               // 高 nibble
            }
        }
    }
    return buf;
}

// VQ2 布局（与 matvec_vq2_ref 完全对齐）：
// [码本 256×4 fp16 = 2048B][索引区 out_dim × (in_dim/4) uint8]。
// 码本与索引独立随机（块 VQ 的合成数据：校验的是"查表+乘加"，不是量化质量）。
static std::vector<uint8_t> pack_vq2(int out_dim, int in_dim, Rng &rng) {
    const size_t cb_bytes = 2048;
    const size_t idx_bytes = static_cast<size_t>(out_dim) * (in_dim / 4);
    std::vector<uint8_t> buf(cb_bytes + idx_bytes);
    for (size_t i = 0; i < 256 * 4; ++i) {
        const uint16_t h = float_to_half(rng.next_f32());
        std::memcpy(buf.data() + i * 2, &h, 2);
    }
    for (size_t i = 0; i < idx_bytes; ++i) {
        buf[cb_bytes + i] = static_cast<uint8_t>(rng.next_u32() & 0xFF);
    }
    return buf;
}

// 各族权重的字节数（用于带宽统计，与加载器/格式的口径一致）
static size_t weight_bytes(const std::string &family, int out_dim, int in_dim, int gs) {
    if (family == "f32") return static_cast<size_t>(out_dim) * in_dim * 4;
    if (family == "f16") return static_cast<size_t>(out_dim) * in_dim * 2;
    if (family == "i4") {
        const int groups_per_row = (in_dim + gs - 1) / gs;
        return static_cast<size_t>(out_dim) * groups_per_row * (4 + gs / 2);
    }
    // vq2：码本 2048B + 每权重 1 字节索引
    return 2048 + static_cast<size_t>(out_dim) * (in_dim / 4);
}

// ============================================================================
// 已注册实现枚举 / 过滤
// ============================================================================
static std::vector<std::string> split_csv(const char *s) {
    std::vector<std::string> out;
    if (!s) return out;
    std::string cur;
    for (const char *p = s; *p; ++p) {
        if (*p == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (*p != ' ') cur.push_back(*p);  // 跳过空格：注册表用", "分隔
    }
    if (!cur.empty()) out.push_back(cur);
    std::sort(out.begin(), out.end());
    // "ref" 恒置首：它是校验锚 + 加速比基准，必须先跑出耗时
    auto it = std::find(out.begin(), out.end(), "ref");
    if (it != out.end() && it != out.begin()) {
        std::rotate(out.begin(), it, it + 1);
    }
    return out;
}

// ============================================================================
// 计时：先跑一次估单次耗时，按 --budget-ms 折算迭代数（带上下限），
// 再预热、正式计时。返回平均每次调用的秒数。
// ============================================================================
struct Timer {
    double budget_s;
    int min_iters;
    int max_iters;
};

template <typename Fn>
static double time_kernel(Fn &&fn, const Timer &t, int *out_iters) {
    // 1. 单次试跑估计耗时
    auto t0 = Clock::now();
    fn();
    auto t1 = Clock::now();
    const double one_s = std::chrono::duration<double>(t1 - t0).count();

    int iters = static_cast<int>(t.budget_s / std::max(one_s, 1e-6));
    iters = std::max(t.min_iters, std::min(t.max_iters, iters));

    // 2. 预热（多线程变体的线程池在这里建好，不进计时窗口）
    const int warmup = std::max(2, iters / 10);
    for (int i = 0; i < warmup; ++i) fn();

    // 3. 正式计时
    t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    t1 = Clock::now();
    *out_iters = iters;
    return std::chrono::duration<double>(t1 - t0).count() / iters;
}

// ============================================================================
// 校验分级：相对 ref 的最大相对误差
//   PASS —— ≤5e-4：浮点累加顺序级差异（ref 用 double，变体用 float/NEON）
//   TOL  —— ≤5e-2：有损但可接受（如 W4A8 激活量化）
//   FAIL —— 超出：实现错误
// ============================================================================
static const char *verify_class(double rel) {
    if (rel <= 5e-4) return "PASS";
    if (rel <= 5e-2) return "TOL ";
    return "FAIL";
}

// ============================================================================
// 单条结果记录（供表格打印 + CSV/JSON 落盘）
// ============================================================================
struct BenchRow {
    std::string family;
    std::string shape;
    int out_dim = 0, in_dim = 0;
    std::string impl;
    std::string verdict = "?";
    double max_rel = 0.0;
    double s_per_call = 0.0;
    int iters = 0;
    double gb_s = 0.0;
    double gflop_s = 0.0;
    double speedup = 1.0;
};

static void print_row(const BenchRow &r) {
    std::printf("  %-16s %-4s rel=%-9.2e %9.3f ms %8.2f GB/s %8.2f GFLOP/s %6.2fx (%d it)\n",
                r.impl.c_str(), r.verdict.c_str(), r.max_rel,
                r.s_per_call * 1e3, r.gb_s, r.gflop_s, r.speedup, r.iters);
}

// ============================================================================
// 主流程
// ============================================================================
int main(int argc, char **argv) {
    // ---------- 默认参数 ----------
    std::string family = "all";        // all | vq2 | i4 | f16 | f32
    std::vector<Shape> shapes = kQwen05bShapes;
    int group_size = 128;              // i4 分组
    bool big = false;                  // f32 是否测 lm_head 大形状
    bool list_only = false;
    bool allow_cuda = false;
    bool no_verify = false;
    Timer timer{0.25, 3, 5000};        // 0.25s 预算/实现×形状
    std::string csv_path, json_path;
    std::vector<std::string> impl_filter;   // --impls：精确名单
    int custom_out = 0, custom_in = 0;

    // ---------- 参数解析 ----------
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char *what) -> const char * {
            if (i + 1 >= argc) {
                std::printf("missing value for %s\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--family") family = need("--family");
        else if (a == "--preset") {
            std::string p = need("--preset");
            if (p == "qwen05b") shapes = kQwen05bShapes;
            else if (p == "qwen3_06b") shapes = kQwen3_06bShapes;
            else { std::printf("unknown preset: %s (qwen05b | qwen3_06b)\n", p.c_str()); return 2; }
        }
        else if (a == "--out") custom_out = std::atoi(need("--out"));
        else if (a == "--in") custom_in = std::atoi(need("--in"));
        else if (a == "--group-size") group_size = std::atoi(need("--group-size"));
        else if (a == "--budget-ms") timer.budget_s = std::atof(need("--budget-ms")) / 1e3;
        else if (a == "--max-iters") timer.max_iters = std::atoi(need("--max-iters"));
        else if (a == "--impls") impl_filter = split_csv(need("--impls"));
        else if (a == "--csv") csv_path = need("--csv");
        else if (a == "--json") json_path = need("--json");
        else if (a == "--big") big = true;
        else if (a == "--list") list_only = true;
        else if (a == "--allow-cuda") allow_cuda = true;
        else if (a == "--no-verify") no_verify = true;
        else {
            std::printf("unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    if (custom_out > 0 && custom_in > 0) {
        shapes = {{"custom", custom_out, custom_in, false}};
    }

    // ---------- 族的枚举（vq2 优先：当前优化重点） ----------
    struct FamilySpec {
        std::string name;
        const char *(*available)();
        bool (*set_by_name)(const char *);
    };
    std::vector<FamilySpec> families;
    auto want = [&](const char *f) { return family == "all" || family == f; };
    if (want("vq2")) families.push_back({"vq2", available_matvec_vq2_impls, set_matvec_vq2_impl_by_name});
    if (want("i4")) families.push_back({"i4", available_matvec_i4_impls, set_matvec_i4_impl_by_name});
    if (want("f16")) families.push_back({"f16", available_matvec_f16_impls, set_matvec_f16_impl_by_name});
    if (want("f32")) families.push_back({"f32", available_matvec_impls, set_matvec_impl_by_name});
    if (families.empty()) {
        std::printf("unknown --family: %s (want all|vq2|i4|f16|f32)\n", family.c_str());
        return 2;
    }

    // ---------- --list：只打印注册表 ----------
    if (list_only) {
        for (auto &f : families) {
            std::printf("%-4s: %s\n", f.name.c_str(), f.available());
        }
        return 0;
    }

    // 实现名过滤器：--impls 精确名单；否则排除子串 "cuda"（除非 --allow-cuda）
    auto impl_ok = [&](const std::string &name) {
        if (!impl_filter.empty()) {
            return std::find(impl_filter.begin(), impl_filter.end(), name) != impl_filter.end();
        }
        if (!allow_cuda && name.find("cuda") != std::string::npos) return false;
        return true;
    };

    std::printf("# bench_kernels | shapes=%s | i4 group_size=%d | budget=%.0fms/impl | impls filter=%s\n",
                custom_out ? "custom" : "qwen05b", group_size, timer.budget_s * 1e3,
                impl_filter.empty() ? (allow_cuda ? "all" : "all,-cuda") : "--impls");

    std::vector<BenchRow> rows;

    for (auto &fam : families) {
        const auto impls_all = split_csv(fam.available());
        std::vector<std::string> impls;
        for (auto &n : impls_all) if (impl_ok(n)) impls.push_back(n);
        if (impls.empty()) {
            std::printf("\n=== matvec %s: no usable impls (registered: %s) ===\n",
                        fam.name.c_str(), fam.available());
            continue;
        }

        // 保证 ref 一定在场：它是校验基准 + 加速比基准
        if (!fam.set_by_name("ref")) {
            std::printf("\n=== matvec %s: 'ref' not registered, skip ===\n", fam.name.c_str());
            continue;
        }

        // 权重缓冲常驻区：sdot2 系按权重指针缓存预计算结果（g_pre_cache），
        // 若每个形状用完即释放，分配器复用地址会造成旧缓存命中→错误输出。
        // 真实运行时权重常驻，故此处同样让所有形状的缓冲存活，地址互不相同。
        std::vector<std::vector<uint8_t>> keepalive;
        std::vector<uint8_t> w_bytes;       // i4 / vq2 packed（当前形状）

        for (const auto &sh : shapes) {
            if (sh.big && fam.name == "f32" && !big) {
                std::printf("\n=== matvec %s | %s %dx%d: skipped (f32 big shape, use --big) ===\n",
                            fam.name.c_str(), sh.name, sh.out_dim, sh.in_dim);
                continue;
            }
            if (fam.name == "vq2" && sh.in_dim % 4 != 0) continue;  // 布局硬约束

            const int OUT = sh.out_dim, IN = sh.in_dim;
            Rng rng(0x5EED + OUT * 1315423911u + IN);

            // ---- 输入 x：族间共享同一随机向量 ----
            std::vector<float> x(IN);
            { Rng xrng(0xABCD1234u); for (auto &v : x) v = xrng.next_f32(); }

            // ---- 按族构造权重缓冲 ----
            std::vector<float> base;            // f32 基础矩阵（f32/f16/i4 用）
            std::vector<uint16_t> w_f16;
            if (fam.name == "vq2") {
                if (!w_bytes.empty()) keepalive.push_back(std::move(w_bytes));
                w_bytes = pack_vq2(OUT, IN, rng);
            } else {
                base = gen_matrix(static_cast<size_t>(OUT) * IN, rng);
                if (fam.name == "f16") w_f16 = to_f16(base);
                if (fam.name == "i4") {
                    // 旧形状的缓冲移入常驻区保活，防止地址复用命中预计算缓存
                    if (!w_bytes.empty()) keepalive.push_back(std::move(w_bytes));
                    w_bytes = pack_i4(base, OUT, IN, group_size);
                }
            }
            const size_t wbytes = weight_bytes(fam.name, OUT, IN, group_size);

            // 统一调用器：按族把 (w, x, y) 送进 dispatch 通用入口
            std::vector<float> y(OUT);
            auto run_current = [&]() {
                if (fam.name == "f32") matvec_f32(base.data(), x.data(), y.data(), OUT, IN);
                else if (fam.name == "f16") matvec_f16(w_f16.data(), x.data(), y.data(), OUT, IN);
                else if (fam.name == "i4") matvec_i4(w_bytes.data(), x.data(), y.data(), OUT, IN, group_size);
                else matvec_vq2(w_bytes.data(), x.data(), y.data(), OUT, IN);
                soak(y.data(), OUT);
            };

            // ---- ref 基准输出（校验锚）：每个形状都要重新锚定，
            // 上一形状的最后一个实现还挂在 dispatch 的"当前实现"上 ----
            std::vector<float> y_ref;
            if (!no_verify) {
                fam.set_by_name("ref");
                run_current();
                y_ref = y;
            }

            std::printf("\n=== matvec %s | %s %dx%d | weight %.1f MB ===\n",
                        fam.name.c_str(), sh.name, OUT, IN, wbytes / 1048576.0);

            double ref_s = 0.0;
            for (size_t ii = 0; ii < impls.size(); ++ii) {
                const std::string &name = impls[ii];
                if (!fam.set_by_name(name.c_str())) {
                    std::printf("  %-16s skip (set_by_name failed)\n", name.c_str());
                    continue;
                }

                BenchRow row;
                row.family = fam.name;
                row.shape = sh.name;
                row.out_dim = OUT;
                row.in_dim = IN;
                row.impl = name;

                // ---- 校验 ----
                if (!no_verify) {
                    run_current();
                    double max_abs = 0.0, denom = 1e-30;
                    for (int k = 0; k < OUT; ++k) {
                        max_abs = std::max(max_abs, static_cast<double>(std::fabs(y[k] - y_ref[k])));
                        denom = std::max(denom, static_cast<double>(std::fabs(y_ref[k])));
                    }
                    row.max_rel = max_abs / denom;
                    row.verdict = verify_class(row.max_rel);
                } else {
                    row.verdict = "-";
                }

                // ---- 测速 ----
                row.s_per_call = time_kernel(run_current, timer, &row.iters);
                row.gb_s = (wbytes + static_cast<size_t>(OUT + IN) * 4) / row.s_per_call / 1e9;
                row.gflop_s = 2.0 * OUT * IN / row.s_per_call / 1e9;
                if (name == "ref") ref_s = row.s_per_call;
                row.speedup = (ref_s > 0) ? ref_s / row.s_per_call : 1.0;

                print_row(row);
                rows.push_back(row);
            }
        }
    }

    // ---------- BiIP 激活旋转（VQ2 旋转推理路径开销，带宽型小算子） ----------
    // 形状取真实部署值：in_dim ∈ {hidden 1024, q_dim 2048, intermediate 3072}，
    // block 一律 256（= find_hadamard_block_size 对这三个维度的取值）。
    // 旋转模型每层 7 个子层各旋一次且不能融合，故逐调用开销要乘 7×n_layers
    // 才是每 token 的真实代价——这里直接折算打印。
    if (want("vq2")) {
        const int block = 256;
        // Qwen3-0.6B 每层 7 个子层的输入维度：q/k/v/gate/up = 1024, o = 2048, down = 3072
        const int per_layer[] = {1024, 1024, 1024, 2048, 1024, 1024, 3072};
        const int n_layers = 28;
        std::printf("\n=== biip_rotate_activation | block=%d ===\n", block);
        for (const char *impl : {"ref", "neon"}) {
            if (!set_ops_impl_by_name(impl)) {
                std::printf("  [skip] 无 '%s' ops 变体\n", impl);
                continue;
            }
            double per_token_us = 0.0;
            for (int dim : {1024, 2048, 3072}) {
                std::vector<float> x(dim), yr(dim), sign(dim), scale(dim);
                Rng rng(0xB11B);
                for (auto &v : x) v = rng.next_f32();
                for (auto &v : sign) v = (rng.next_u32() & 1) ? 1.0f : -1.0f;
                for (auto &v : scale) v = 0.5f + 0.5f * std::fabs(rng.next_f32());
                auto fn = [&]() {
                    biip_rotate_activation(x.data(), yr.data(), dim, scale.data(), sign.data(),
                                           block);
                    soak(yr.data(), dim);
                };
                int iters = 0;
                const double s = time_kernel(fn, timer, &iters);
                const double gb = (4.0 * dim * 4) / s / 1e9;  // 读 x+scale+sign、写 y
                int calls = 0;
                for (int d : per_layer) calls += (d == dim) ? 1 : 0;
                calls *= n_layers;
                per_token_us += s * 1e6 * calls;
                std::printf("  %-5s dim=%-5d %9.4f us %8.2f GB/s (%d it, 每 token %d 次)\n",
                            impl, dim, s * 1e6, gb, iters, calls);
            }
            std::printf("  %-5s -> 每 token 旋转总开销 %.3f ms（%d 层 x 7 子层）\n",
                        impl, per_token_us / 1000.0, n_layers);
        }
        set_ops_impl_by_name("ref");
    }

    // ---------- CSV 落盘 ----------
    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        if (!f) {
            std::printf("cannot open %s for write\n", csv_path.c_str());
            return 1;
        }
        f << "family,shape,out_dim,in_dim,impl,verdict,max_rel,ms_per_call,gb_s,gflop_s,speedup,iters\n";
        for (auto &r : rows) {
            char line[512];
            std::snprintf(line, sizeof(line),
                          "%s,%s,%d,%d,%s,%s,%.6e,%.6f,%.3f,%.3f,%.3f,%d\n",
                          r.family.c_str(), r.shape.c_str(), r.out_dim, r.in_dim,
                          r.impl.c_str(), r.verdict.c_str(), r.max_rel,
                          r.s_per_call * 1e3, r.gb_s, r.gflop_s, r.speedup, r.iters);
            f << line;
        }
        std::printf("\n[csv] %s (%zu rows)\n", csv_path.c_str(), rows.size());
    }

    // ---------- JSON 落盘（极简：行数组） ----------
    if (!json_path.empty()) {
        std::ofstream f(json_path);
        if (!f) {
            std::printf("cannot open %s for write\n", json_path.c_str());
            return 1;
        }
        f << "[\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            auto &r = rows[i];
            char line[640];
            std::snprintf(line, sizeof(line),
                          "  {\"family\":\"%s\",\"shape\":\"%s\",\"out\":%d,\"in\":%d,"
                          "\"impl\":\"%s\",\"verdict\":\"%s\",\"max_rel\":%.6e,"
                          "\"ms\":%.6f,\"gb_s\":%.3f,\"gflop_s\":%.3f,\"speedup\":%.3f}%s\n",
                          r.family.c_str(), r.shape.c_str(), r.out_dim, r.in_dim,
                          r.impl.c_str(), r.verdict.c_str(), r.max_rel,
                          r.s_per_call * 1e3, r.gb_s, r.gflop_s, r.speedup,
                          (i + 1 < rows.size()) ? "," : "");
            f << line;
        }
        f << "]\n";
        std::printf("[json] %s (%zu rows)\n", json_path.c_str(), rows.size());
    }

    // ---------- 预期不匹配的说明（契约差异，不是实现错误） ----------
    {
        bool any_sdot5 = false;
        for (auto &r : rows) {
            if (r.verdict == "FAIL" && r.impl.find("sdot5") != std::string::npos) any_sdot5 = true;
        }
        if (any_sdot5) {
            std::printf("\n[note] sdot5/sdot5_mt 是对称量化专用变体（zero 恒 8），本基准用非对称随机权重，\n"
                        "       FAIL 为预期；评测对称模型请用 --symmetric 模型 + tools/bench.py 端到端核对。\n");
        }
    }

    std::printf("\n[sink] %g\n", static_cast<double>(g_sink));
    return 0;
}
