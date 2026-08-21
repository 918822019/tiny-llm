// ============================================================================
// matvec_i4_neon_mt.cpp — INT4 weight-only matvec：NEON + 常驻线程池行切分
// ============================================================================
// 本文件在 matvec_i4_neon（单线程 NEON 版）基础上叠加多线程行切分。
//
// 布局与 matvec_i4_neon 完全一致（每组 [scale_fp16 | zero_fp16 | packed_uint4]，
// 低 nibble 在前，(q-zero)×scale 反量化）；本文件只加"多核并行按输出行切分"，
// 线程池结构与 f16 neon_mt_kv_nt 的 RowPool 相同（spin barrier、常驻 worker）。
//
// 背景（优化日志 i4-hqq-android）：i4 单线程 neon 层部分实测仅 ~3 GB/s 有效
// 带宽（nibble unpack 吃满单核），多线程行切分把 unpack 摊到多核是本变体的目的。
// group_size 运行时传入（v2 header quant_group_size），pool 按调用存储。
//
// 仅 aarch64 构建注册；其他平台编译为空翻译单元。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

// ========================================================================
// dot_group_i4_neon() — 处理单行的一个 group（与 matvec_i4_neon 相同实现）
// ========================================================================
// 功能：NEON 加速的组内反量化点积
inline float dot_group_i4_neon(const uint8_t *packed, const float *x,
                               float scale, float zero, int group_elems) {
    const float32x4_t v_scale = vdupq_n_f32(scale);         // scale 向量
    const float32x4_t v_neg_zs = vdupq_n_f32(-zero * scale); // -zero*scale 向量

    float32x4_t acc0 = vdupq_n_f32(0.0f); // 累加器 0
    float32x4_t acc1 = vdupq_n_f32(0.0f); // 累加器 1
    float32x4_t acc2 = vdupq_n_f32(0.0f); // 累加器 2
    float32x4_t acc3 = vdupq_n_f32(0.0f); // 累加器 3

    const uint8x16_t mask_lo = vdupq_n_u8(0x0F); // 低 nibble 掩码

    int i = 0;
    const int n32 = group_elems & ~31; // 主循环边界

    for (; i < n32; i += 32) {
        // 加载 16 字节 packed → 拆分高低 nibble → 交错重排
        uint8x16_t raw = vld1q_u8(packed + i / 2);
        uint8x16_t lo8 = vandq_u8(raw, mask_lo);       // 偶数下标 uint4
        uint8x16_t hi8 = vshrq_n_u8(raw, 4);           // 奇数下标 uint4
        uint8x16x2_t zipped = vzipq_u8(lo8, hi8);      // 交错为顺序排列

        // uint8 → uint16 零扩展
        uint16x8_t w16_0 = vmovl_u8(vget_low_u8(zipped.val[0]));
        uint16x8_t w16_1 = vmovl_u8(vget_high_u8(zipped.val[0]));
        uint16x8_t w16_2 = vmovl_u8(vget_low_u8(zipped.val[1]));
        uint16x8_t w16_3 = vmovl_u8(vget_high_u8(zipped.val[1]));

        // uint16 → uint32 → float32 → 反量化 → FMA 累加（8 组各 4 个元素）
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_0)));
        float32x4_t x0 = vld1q_f32(x + i);
        f0 = vfmaq_f32(v_neg_zs, f0, v_scale); // 反量化
        acc0 = vfmaq_f32(acc0, f0, x0);          // 乘累加

        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_0)));
        float32x4_t x1 = vld1q_f32(x + i + 4);
        f1 = vfmaq_f32(v_neg_zs, f1, v_scale);
        acc1 = vfmaq_f32(acc1, f1, x1);

        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_1)));
        float32x4_t x2 = vld1q_f32(x + i + 8);
        f2 = vfmaq_f32(v_neg_zs, f2, v_scale);
        acc2 = vfmaq_f32(acc2, f2, x2);

        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_1)));
        float32x4_t x3 = vld1q_f32(x + i + 12);
        f3 = vfmaq_f32(v_neg_zs, f3, v_scale);
        acc3 = vfmaq_f32(acc3, f3, x3);

        float32x4_t f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_2)));
        float32x4_t x4 = vld1q_f32(x + i + 16);
        f4 = vfmaq_f32(v_neg_zs, f4, v_scale);
        acc0 = vfmaq_f32(acc0, f4, x4);

        float32x4_t f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_2)));
        float32x4_t x5 = vld1q_f32(x + i + 20);
        f5 = vfmaq_f32(v_neg_zs, f5, v_scale);
        acc1 = vfmaq_f32(acc1, f5, x5);

        float32x4_t f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_3)));
        float32x4_t x6 = vld1q_f32(x + i + 24);
        f6 = vfmaq_f32(v_neg_zs, f6, v_scale);
        acc2 = vfmaq_f32(acc2, f6, x6);

        float32x4_t f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_3)));
        float32x4_t x7 = vld1q_f32(x + i + 28);
        f7 = vfmaq_f32(v_neg_zs, f7, v_scale);
        acc3 = vfmaq_f32(acc3, f7, x7);
    }

    // 合并 + 横向归约
    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float total = vaddvq_f32(vaddq_f32(sum01, sum23));

    // 标量尾段
    for (; i < group_elems; ++i) {
        const int byte_idx = i / 2;
        uint8_t val = (i % 2 == 0) ? (packed[byte_idx] & 0x0F) : ((packed[byte_idx] >> 4) & 0x0F);
        total += (static_cast<float>(val) - zero) * scale * x[i];
    }

    return total;
}

constexpr int kGroupHeader = 4; // 每组头部字节数

// ========================================================================
// dot_row_i4() — 单行点积：遍历该行的所有 group
// ========================================================================
inline float dot_row_i4(const uint8_t *row, const float *x, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total_bytes = kGroupHeader + group_size / 2;
    float acc = 0.0f;
    int col = 0;
    for (int g = 0; g < groups_per_row; ++g) {
        const uint8_t *gp = row + g * group_total_bytes; // 组起始地址
        // 安全读取 fp16 scale/zero
        uint16_t scale_h, zero_h;
        std::memcpy(&scale_h, gp, 2);
        std::memcpy(&zero_h, gp + 2, 2);
        const float scale = half_to_float(scale_h);
        const float zero = half_to_float(zero_h);
        const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);
        acc += dot_group_i4_neon(gp + kGroupHeader, x + col, scale, zero, group_elems);
        col += group_size;
    }
    return acc;
}

// 自旋等待原子计数器到达目标值
inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
    int spins = 0;
    while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
            __builtin_arm_yield(); // ARM YIELD 指令
        } else {
            std::this_thread::yield();
        }
    }
}

// 并行度：TINYQWEN_MT_THREADS 覆盖；默认 = 硬件核数（上限 16）
inline int default_parallelism_i4() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolI4 — 常驻行切分线程池（结构同 f16 neon_mt_kv_nt 的 RowPool）
// ========================================================================
struct RowPoolI4 {
    const uint8_t *w = nullptr; // INT4 packed 权重
    const float *x = nullptr;   // 输入向量
    float *y = nullptr;         // 输出向量
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 64;        // 量化分组大小（运行时传入）

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolI4() {
        const int p = default_parallelism_i4();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ~RowPoolI4() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t: workers) t.join();
    }

    void worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_chunk(idx);
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    // 第 idx 块行区间上的逐行 INT4 点积
    void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        const int groups_per_row = (in_dim + group_size - 1) / group_size;
        const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);
        for (int o = begin; o < end; ++o) {
            y[o] = dot_row_i4(w + static_cast<size_t>(o) * row_bytes, x, in_dim, group_size);
        }
    }

    // fork-join 入口
    void run(const uint8_t *w_, const float *x_, float *y_, int out_dim_, int in_dim_,
             int group_size_) {
        w = w_; x = x_; y = y_;
        out_dim = out_dim_; in_dim = in_dim_; group_size = group_size_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
    }
};

// Meyers singleton
RowPoolI4 &pool_i4() {
    static RowPoolI4 p;
    return p;
}

// 粒度阈值（按元素数）：小矩阵单线程，省同步开销
constexpr std::size_t kMinParallelElems = 262144;

// ========================================================================
// matvec_i4_neon_mt() — INT4 matvec NEON + 多线程入口
// ========================================================================
void matvec_i4_neon_mt(const uint8_t *w, const float *x, float *y,
                       int out_dim, int in_dim, int group_size) {
    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElems ||
        std::thread::hardware_concurrency() <= 1) {
        // 单线程路径
        const int groups_per_row = (in_dim + group_size - 1) / group_size;
        const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);
        for (int o = 0; o < out_dim; ++o) {
            y[o] = dot_row_i4(w + static_cast<size_t>(o) * row_bytes, x, in_dim, group_size);
        }
        return;
    }
    pool_i4().run(w, x, y, out_dim, in_dim, group_size);
}

} // namespace

// 自注册进 dispatch：--matvec-impl neon_mt（i4 注册表）
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_neon_mt, "neon_mt");

} // namespace tinyqwen

#endif // __aarch64__
