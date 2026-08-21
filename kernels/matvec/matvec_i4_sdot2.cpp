// ============================================================================
// matvec_i4_sdot2.cpp — INT4 weight-only matvec：y = W @ x
//                  W4A8 SDOT v2（优化版）+ 多线程版
// ============================================================================
// 本文件实现两种变体：单线程 "sdot2" 和多线程 "sdot2_mt"。
//
// 在 sdot 基础上两项优化：
//   ① 预计算 scale/zero 为 f32 数组（按权重指针缓存，首次调用时 unpack）
//     消除 per-group 的 2×memcpy + 2×half_to_float + 1×FSUB = 5 inst/group。
//     对 Qwen2.5-0.5B (group=64) 省 ~39M inst/token (~10%)。
//   ② 2-row 并行内循环：2 条独立 SDOT 链 + 共享激活加载
//     每迭代省 1 条 vld1q_s8（激活共享），2 条独立链提高 ILP。
//
// 布局与 sdot 完全一致（每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(G/2 B)]），
// 只是 runtime 侧预计算了 float32 scale 和 (zero-8) 到独立数组。
// group_size 必须是 16 的倍数（SDOT 每次 16 元素）。
//
// 条件编译：需要 __aarch64__ && __ARM_FEATURE_DOTPROD。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>          // std::mutex（保护预计算缓存）
#include <thread>
#include <unordered_map>  // 预计算缓存的哈希表
#include <vector>

namespace tinyqwen {
namespace {

constexpr int kGroupHeader = 4;    // 每组头部字节数
constexpr int kMaxInDim = 8192;    // 支持的最大输入维度

// 全局 scratch 缓冲区（串行调用，静态安全）
int8_t  g_xq2[kMaxInDim];           // int8 量化后的激活
int32_t g_xq2_prefix[kMaxInDim + 1]; // 前缀和

// ========================================================================
// quantize_x_i8_v2() — 对称 int8 量化 x，并填前缀和
// ========================================================================
// 功能：与 sdot 版的 quantize_x_i8 相同逻辑，使用独立的 scratch 缓冲区
float quantize_x_i8_v2(const float *x, int in_dim) {
    const int n = in_dim < kMaxInDim ? in_dim : kMaxInDim;
    // 找绝对值最大值
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    // 计算缩放因子
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale_x;
    g_xq2_prefix[0] = 0;
    // 逐元素量化 + 填前缀和
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v); // clamp
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f); // 四舍五入
        g_xq2[i] = static_cast<int8_t>(q);
        g_xq2_prefix[i + 1] = g_xq2_prefix[i] + q;
    }
    return scale_x;
}

// ========================================================================
// PrecomputedI4 — 预计算的 scale/zero f32 数组
// ========================================================================
// 将 interleaved [scale_fp16|zero_fp16|packed] 中的 scale/zero 解包成
// 连续 f32 数组，消除 per-group 的 memcpy + half_to_float + FSUB 开销。
struct PrecomputedI4 {
    std::vector<float> scales;      // [out_dim * groups_per_row]：每组的 scale（fp32）
    std::vector<float> zeros_m8;    // [out_dim * groups_per_row]：每组的 (zero - 8)（fp32）
    int out_dim = 0;                // 行数
    int groups_per_row = 0;         // 每行的组数
};

// 全局预计算缓存：key = 权重指针地址，value = 预计算结果
std::unordered_map<const uint8_t *, PrecomputedI4> g_pre_cache;
std::mutex g_pre_mutex; // 保护缓存的互斥锁

// ========================================================================
// get_or_precompute() — 获取或创建预计算数据
// ========================================================================
// 功能：首次见到某权重指针时解包 scale/zero；后续调用命中缓存零开销。
// 参数：w — 权重指针, out_dim/in_dim/group_size — 矩阵维度
// 返回值：预计算数据的常量引用
const PrecomputedI4 &get_or_precompute(const uint8_t *w, int out_dim, int in_dim,
                                        int group_size) {
    std::lock_guard<std::mutex> lock(g_pre_mutex); // 加锁保护缓存
    auto it = g_pre_cache.find(w); // 查找缓存
    if (it != g_pre_cache.end()) return it->second; // 命中则直接返回

    // 未命中：执行预计算
    const int gpr = (in_dim + group_size - 1) / group_size; // 每行组数
    const int group_bytes = kGroupHeader + group_size / 2;   // 每组字节数
    PrecomputedI4 p;
    p.out_dim = out_dim;
    p.groups_per_row = gpr;
    p.scales.resize(static_cast<size_t>(out_dim) * gpr);
    p.zeros_m8.resize(static_cast<size_t>(out_dim) * gpr);

    // 遍历所有行和组，解包 fp16 scale/zero 到 fp32 数组
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = w + static_cast<size_t>(o) * gpr * group_bytes;
        for (int g = 0; g < gpr; ++g) {
            const uint8_t *gp = row + static_cast<size_t>(g) * group_bytes;
            uint16_t sh, zh;
            std::memcpy(&sh, gp, 2);       // 读 scale（fp16）
            std::memcpy(&zh, gp + 2, 2);   // 读 zero（fp16）
            p.scales[o * gpr + g] = half_to_float(sh);           // fp16 → fp32
            p.zeros_m8[o * gpr + g] = half_to_float(zh) - 8.0f; // fp16 → fp32 并减 8
        }
    }
    // 插入缓存并返回
    auto [it2, _] = g_pre_cache.emplace(w, std::move(p));
    return it2->second;
}

// ========================================================================
// dot_row_i4_sdot2() — 单行点积（预计算版）
// ========================================================================
// 功能：用预计算的 scale/zero 数组加速的单行 SDOT 点积
// 参数：
//   row       — 该行的 packed 权重起始地址
//   pre       — 预计算数据
//   row_idx   — 行索引（用于在预计算数组中定位）
//   in_dim    — 输入维度
//   group_size — 量化分组大小
//   scale_x   — 激活的量化缩放因子
// 返回值：该行的输出值（fp32）
float dot_row_i4_sdot2(const uint8_t *row, const PrecomputedI4 &pre, int row_idx,
                       int in_dim, int group_size, float scale_x) {
    const int gpr = pre.groups_per_row;
    const int group_bytes = kGroupHeader + group_size / 2;
    float acc = 0.0f;
    int col = 0;

    const uint8x16_t v8 = vdupq_n_u8(8);       // 常量 8
    const uint8x8_t mask_lo = vdup_n_u8(0x0F); // 低 nibble 掩码

    for (int g = 0; g < gpr; ++g) {
        // 从预计算数组直接读 f32 — 无 memcpy / half_to_float / FSUB
        const float scale_w = pre.scales[row_idx * gpr + g];
        const float zero_m8 = pre.zeros_m8[row_idx * gpr + g]; // 已减 8
        const float A = scale_w * scale_x;  // DOT_g 的系数
        const float C = A * zero_m8;        // XQSUM 的系数

        const uint8_t *packed = row + static_cast<size_t>(g) * group_bytes + kGroupHeader;
        const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

        // SDOT 累加器
        int32x4_t acc_dot = vdupq_n_s32(0);
        int i = 0;
        const int n16 = group_elems & ~15;
        for (; i < n16; i += 16) {
            // 加载 packed → 拆分 nibble → 交错重排 → 转有符号
            uint8x8_t raw = vld1_u8(packed + i / 2);
            uint8x8_t lo = vand_u8(raw, mask_lo);
            uint8x8_t hi = vshr_n_u8(raw, 4);
            uint8x8x2_t z = vzip_u8(lo, hi);
            uint8x16_t q_u = vcombine_u8(z.val[0], z.val[1]);
            int8x16_t q_s = vreinterpretq_s8_u8(vsubq_u8(q_u, v8)); // q-8
            // 加载 int8 量化激活
            int8x16_t xq = vld1q_s8(g_xq2 + col + i);
            // SDOT：16 个 int8×int8 → 4 路 int32 累加
            acc_dot = vdotq_s32(acc_dot, q_s, xq);
        }
        // 横向归约
        int dot_g = vaddvq_s32(acc_dot);

        // 标量尾部
        for (; i < group_elems; ++i) {
            const int byte_idx = i / 2;
            const int q = (i % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                       : ((packed[byte_idx] >> 4) & 0x0F);
            dot_g += (q - 8) * static_cast<int>(g_xq2[col + i]);
        }

        // 用前缀和 O(1) 查询组内 x_q 之和
        const int xqsum_g = g_xq2_prefix[col + group_elems] - g_xq2_prefix[col];
        // 还原浮点结果
        acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum_g);
        col += group_size;
    }
    return acc;
}

// ========================================================================
// dot_2rows_i4_sdot2() — 2-row 并行点积：2 条独立 SDOT 链 + 共享激活加载
// ========================================================================
// 功能：同时计算两行的 SDOT 点积，共享激活加载以节省带宽
// 参数：
//   row_a, row_b — 两行的 packed 权重起始地址
//   pre          — 预计算数据
//   row_idx_a/b  — 两行的索引
//   in_dim, group_size, scale_x — 维度和量化参数
//   out_a, out_b — 输出指针
// 说明：每迭代省 1 条 vld1q_s8（激活共享），2 条独立 SDOT 链提高 ILP。
void dot_2rows_i4_sdot2(const uint8_t *row_a, const uint8_t *row_b,
                        const PrecomputedI4 &pre, int row_idx_a, int row_idx_b,
                        int in_dim, int group_size, float scale_x,
                        float *out_a, float *out_b) {
    const int gpr = pre.groups_per_row;
    const int group_bytes = kGroupHeader + group_size / 2;
    float acc_a = 0.0f, acc_b = 0.0f; // 两行各自的浮点累加器
    int col = 0;

    const uint8x16_t v8 = vdupq_n_u8(8);
    const uint8x8_t mask_lo = vdup_n_u8(0x0F);

    for (int g = 0; g < gpr; ++g) {
        // 两行的 scale/zero 各自独立（预计算 f32 直接读）
        const float sa = pre.scales[row_idx_a * gpr + g];
        const float za = pre.zeros_m8[row_idx_a * gpr + g];
        const float sb = pre.scales[row_idx_b * gpr + g];
        const float zb = pre.zeros_m8[row_idx_b * gpr + g];
        const float A_a = sa * scale_x, C_a = A_a * za; // 行 A 的系数
        const float A_b = sb * scale_x, C_b = A_b * zb; // 行 B 的系数

        const uint8_t *pk_a = row_a + static_cast<size_t>(g) * group_bytes + kGroupHeader;
        const uint8_t *pk_b = row_b + static_cast<size_t>(g) * group_bytes + kGroupHeader;
        const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

        // 两条独立的 SDOT 累加链
        int32x4_t dot_a = vdupq_n_s32(0);
        int32x4_t dot_b = vdupq_n_s32(0);
        int i = 0;
        const int n16 = group_elems & ~15;
        for (; i < n16; i += 16) {
            // 共享激活加载 — 省 1 条 vld1q_s8（两行共用同一份 x_q）
            int8x16_t xq = vld1q_s8(g_xq2 + col + i);

            // Row A: unpack + SDOT
            uint8x8_t raw_a = vld1_u8(pk_a + i / 2);
            uint8x8_t lo_a = vand_u8(raw_a, mask_lo);
            uint8x8_t hi_a = vshr_n_u8(raw_a, 4);
            uint8x8x2_t z_a = vzip_u8(lo_a, hi_a);
            int8x16_t qs_a = vreinterpretq_s8_u8(vsubq_u8(
                vcombine_u8(z_a.val[0], z_a.val[1]), v8)); // q-8
            dot_a = vdotq_s32(dot_a, qs_a, xq); // SDOT 累加

            // Row B: unpack + SDOT（独立链，不依赖 dot_a）
            uint8x8_t raw_b = vld1_u8(pk_b + i / 2);
            uint8x8_t lo_b = vand_u8(raw_b, mask_lo);
            uint8x8_t hi_b = vshr_n_u8(raw_b, 4);
            uint8x8x2_t z_b = vzip_u8(lo_b, hi_b);
            int8x16_t qs_b = vreinterpretq_s8_u8(vsubq_u8(
                vcombine_u8(z_b.val[0], z_b.val[1]), v8));
            dot_b = vdotq_s32(dot_b, qs_b, xq); // SDOT 累加（共享 xq）
        }
        // 两行各自横向归约
        int sum_a = vaddvq_s32(dot_a);
        int sum_b = vaddvq_s32(dot_b);

        // 标量尾部（两行各处理）
        for (; i < group_elems; ++i) {
            const int bi = i / 2;
            const int qa = (i % 2 == 0) ? (pk_a[bi] & 0x0F) : ((pk_a[bi] >> 4) & 0x0F);
            const int qb = (i % 2 == 0) ? (pk_b[bi] & 0x0F) : ((pk_b[bi] >> 4) & 0x0F);
            const int xq_v = static_cast<int>(g_xq2[col + i]);
            sum_a += (qa - 8) * xq_v;
            sum_b += (qb - 8) * xq_v;
        }

        // 还原浮点结果
        const int xqsum_g = g_xq2_prefix[col + group_elems] - g_xq2_prefix[col];
        const float xs = static_cast<float>(xqsum_g);
        acc_a += A_a * static_cast<float>(sum_a) - C_a * xs;
        acc_b += A_b * static_cast<float>(sum_b) - C_b * xs;
        col += group_size;
    }
    *out_a = acc_a;
    *out_b = acc_b;
}

// ========================================================================
// matvec_i4_sdot2() — 单线程 SDOT v2 matvec 入口
// ========================================================================
void matvec_i4_sdot2(const uint8_t *w, const float *x, float *y,
                     int out_dim, int in_dim, int group_size) {
    const int gpr = (in_dim + group_size - 1) / group_size;
    const int row_bytes = gpr * (kGroupHeader + group_size / 2);

    // 一次性量化激活
    const float scale_x = quantize_x_i8_v2(x, in_dim);
    // 获取或创建预计算数据
    const PrecomputedI4 &pre = get_or_precompute(w, out_dim, in_dim, group_size);

    // 2-row 并行主循环：每次处理两行
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        dot_2rows_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                           w + static_cast<size_t>(o + 1) * row_bytes,
                           pre, o, o + 1, in_dim, group_size, scale_x,
                           &y[o], &y[o + 1]);
    }
    // 奇数行兜底：最后一行单独处理
    if (o < out_dim) {
        y[o] = dot_row_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                                pre, o, in_dim, group_size, scale_x);
    }
}

// ========================================================================
// 多线程版 sdot2_mt
// ========================================================================

// 自旋等待
inline void spin_until_sdot2(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
    int spins = 0;
    while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
#if defined(__aarch64__)
            __builtin_arm_yield();
#endif
        } else {
            std::this_thread::yield();
        }
    }
}

// 默认并行度
inline int default_parallelism_sdot2() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolSdot2 — 常驻行切分线程池（SDOT v2 版）
// ========================================================================
struct RowPoolSdot2 {
    const uint8_t *w = nullptr;
    float *y = nullptr;
    const PrecomputedI4 *pre = nullptr; // 指向预计算数据（master 设置后只读）
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 64;
    float scale_x = 1.0f;

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolSdot2() {
        const int p = default_parallelism_sdot2();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ~RowPoolSdot2() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_sdot2(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_chunk(idx);
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    // 第 idx 块行区间：2-row 并行 + 单行兜底
    void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        const int gpr = pre->groups_per_row;
        const int row_bytes = gpr * (kGroupHeader + group_size / 2);
        // 2-row 并行主循环
        int o = begin;
        for (; o + 1 < end; o += 2) {
            dot_2rows_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                               w + static_cast<size_t>(o + 1) * row_bytes,
                               *pre, o, o + 1, in_dim, group_size, scale_x,
                               &y[o], &y[o + 1]);
        }
        // 奇数行兜底
        if (o < end) {
            y[o] = dot_row_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                                    *pre, o, in_dim, group_size, scale_x);
        }
    }

    // fork-join 入口
    void run(const uint8_t *w_, float *y_, int out_dim_, int in_dim_,
             int group_size_, float scale_x_, const PrecomputedI4 &pre_) {
        w = w_; y = y_; pre = &pre_;
        out_dim = out_dim_; in_dim = in_dim_;
        group_size = group_size_; scale_x = scale_x_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until_sdot2(done_gen, expected_done);
    }
};

// Meyers singleton
RowPoolSdot2 &pool_sdot2() {
    static RowPoolSdot2 p;
    return p;
}

// 粒度阈值
constexpr std::size_t kMinParallelElemsSdot2 = 262144;

// ========================================================================
// matvec_i4_sdot2_mt() — 多线程 SDOT v2 matvec 入口
// ========================================================================
void matvec_i4_sdot2_mt(const uint8_t *w, const float *x, float *y,
                        int out_dim, int in_dim, int group_size) {
    // 一次性量化激活
    const float scale_x = quantize_x_i8_v2(x, in_dim);
    // 获取或创建预计算数据
    const PrecomputedI4 &pre = get_or_precompute(w, out_dim, in_dim, group_size);

    const int gpr = (in_dim + group_size - 1) / group_size;
    const int row_bytes = gpr * (kGroupHeader + group_size / 2);

    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElemsSdot2 ||
        std::thread::hardware_concurrency() <= 1) {
        // 单线程 2-row 路径
        int o = 0;
        for (; o + 1 < out_dim; o += 2) {
            dot_2rows_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                               w + static_cast<size_t>(o + 1) * row_bytes,
                               pre, o, o + 1, in_dim, group_size, scale_x,
                               &y[o], &y[o + 1]);
        }
        if (o < out_dim) {
            y[o] = dot_row_i4_sdot2(w + static_cast<size_t>(o) * row_bytes,
                                    pre, o, in_dim, group_size, scale_x);
        }
        return;
    }
    pool_sdot2().run(w, y, out_dim, in_dim, group_size, scale_x, pre);
}

} // namespace

// 自注册：单线程 "sdot2" + 多线程 "sdot2_mt"
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot2, "sdot2");
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot2_mt, "sdot2_mt");

} // namespace tinyqwen

#endif // __aarch64__ && __ARM_FEATURE_DOTPROD
