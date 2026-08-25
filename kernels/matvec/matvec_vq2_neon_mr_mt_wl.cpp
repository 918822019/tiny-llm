// ============================================================================
// matvec_vq2_neon_mr_mt_wl.cpp — VQ2 matvec：neon_mr_mt + 索引 32 位字加载
// ============================================================================
// matvec_vq2_neon_mr_mt 的归因阶梯下一层。布局/线程池/粒度阈值全部不变，
// 只加一个技术：**索引按 32 位字批量加载**（配套 4 块展开）：
//
//   上一层内循环每块每行要 1 条 ldrb 读索引字节：
//     每块 9 load（1×x4 + 4×idx 字节 + 4×码本）对 4 FMA，
//     3 个 load 端口 → 下限 ~3 周期/块，load 发射是墙。
//   本层把块循环展开 4 倍，每行 4 个连续索引字节用一条 32 位加载读入：
//     每 4 块 24 load（4×x4 + 4×idx 字 + 16×码本）对 16 FMA，
//     → 6 load/块，load 下限降到 ~2 周期/块（-33%）。
//
// 索引区行内天然连续，memcpy 4 字节会被编译为单条 LDR（ARMv8 允许
// 非对齐）；行起点不保证 4 对齐（n_blocks 非 4 倍数时）也安全。
// 块数非 4 倍数的尾段退回逐字节路径。
//
// **尺寸门**（实测教训）：字加载的收益只在 DRAM 流式大形状出现
// （lm_head 151936×896：1.2× 胜出，三轮复现）；中小形状（idx 驻留
// L1/L2）反而被额外的移位/掩码 ALU 与更长的依赖墙拖慢——896×896
// 稳定倒退 2.3×、2048×896 约 2×。故入口按索引区大小分流：
// ≥ kWordLoadMinIdxBytes(8MB) 走字加载，否则退回上一层的逐字节体
// （数值完全相同）。Qwen2.5-0.5B 各投影中仅 lm_head(34MB) 过门。
//
// 数值语义与 neon_mr_mt 完全一致（同一查表 + fp32 累加，行内运算序不变：
// 每行仍按块序 b=0,1,2,... 累加，只是指令调度重排）。
//
// 仅 aarch64 构建注册；其他平台编译为空翻译单元。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VQ2_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float

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

constexpr int kBlockDim = 4;
constexpr int kCodebookEntries = 256;
constexpr int kCodebookBytes = kCodebookEntries * kBlockDim * 2; // 2048B
constexpr int kRows = 4;                                          // 多行并行度
constexpr int kUnroll = 4;                                        // 块展开度（=字宽）

// ========================================================================
// run_range_byte() — 小形状体：逐字节索引加载（与上一层 neon_mr_mt 同款）
// ========================================================================
inline void run_range_byte(const float cb[kCodebookEntries][kBlockDim],
                           const uint8_t *idx, int n_blocks,
                           const float *x, float *y, int begin, int end) {
    int o = begin;
    for (; o + kRows <= end; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), x4);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), x4);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), x4);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), x4);
        }
        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }
    for (; o < end; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[row[b]]), x4);
        }
        y[o] = vaddvq_f32(acc);
    }
}

// ========================================================================
// run_range_wl() — 大形状体：4 行并行 × 4 块展开 × 32 位字索引加载
// ========================================================================
inline void run_range_wl(const float cb[kCodebookEntries][kBlockDim],
                         const uint8_t *idx, int n_blocks,
                         const float *x, float *y, int begin, int end) {
    int o = begin;
    for (; o + kRows <= end; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        int b = 0;
        // 主体：每次 4 块——每行一条 32 位加载取 4 个索引，省 3 条 ldrb
        for (; b + kUnroll <= n_blocks; b += kUnroll) {
            uint32_t w0, w1, w2, w3;
            std::memcpy(&w0, r0 + b, 4); // 单条 LDR（非对齐安全）
            std::memcpy(&w1, r1 + b, 4);
            std::memcpy(&w2, r2 + b, 4);
            std::memcpy(&w3, r3 + b, 4);

            const float32x4_t x0 = vld1q_f32(x + (b + 0) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[w0 & 0xFFu]), x0);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[w1 & 0xFFu]), x0);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[w2 & 0xFFu]), x0);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[w3 & 0xFFu]), x0);

            const float32x4_t x1 = vld1q_f32(x + (b + 1) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[(w0 >> 8) & 0xFFu]), x1);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[(w1 >> 8) & 0xFFu]), x1);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[(w2 >> 8) & 0xFFu]), x1);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[(w3 >> 8) & 0xFFu]), x1);

            const float32x4_t x2 = vld1q_f32(x + (b + 2) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[(w0 >> 16) & 0xFFu]), x2);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[(w1 >> 16) & 0xFFu]), x2);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[(w2 >> 16) & 0xFFu]), x2);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[(w3 >> 16) & 0xFFu]), x2);

            const float32x4_t x3 = vld1q_f32(x + (b + 3) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[w0 >> 24]), x3);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[w1 >> 24]), x3);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[w2 >> 24]), x3);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[w3 >> 24]), x3);
        }
        // 块尾（不足 4 块）：逐字节路径（与上一层同款）
        for (; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), x4);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), x4);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), x4);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), x4);
        }

        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }

    // 尾行（区间内不足 4 行）：单行 + 同款字加载内循环
    for (; o < end; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int b = 0;
        for (; b + kUnroll <= n_blocks; b += kUnroll) {
            uint32_t w;
            std::memcpy(&w, row + b, 4);
            float32x4_t xv = vld1q_f32(x + (b + 0) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[w & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 1) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[(w >> 8) & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 2) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[(w >> 16) & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 3) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[w >> 24]), xv);
        }
        for (; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[row[b]]), x4);
        }
        y[o] = vaddvq_f32(acc);
    }
}

// 码本 [K, d] fp16 → fp32（与上一层同款）
inline void load_codebook(const uint8_t *w, float cb[kCodebookEntries][kBlockDim]) {
    for (int k = 0; k < kCodebookEntries; ++k) {
        for (int j = 0; j < kBlockDim; ++j) {
            uint16_t h;
            std::memcpy(&h, w + (k * kBlockDim + j) * 2, 2);
            cb[k][j] = half_to_float(h);
        }
    }
}

inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
    int spins = 0;
    while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
            __builtin_arm_yield();
        } else {
            std::this_thread::yield();
        }
    }
}

inline int default_parallelism_vq2_wl() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolVQ2Wl — 常驻行切分线程池（结构同 RowPoolVQ2，内循环换成字加载版）
// ========================================================================
struct RowPoolVQ2Wl {
    float cb[kCodebookEntries][kBlockDim];
    const uint8_t *idx_base = nullptr;
    const float *x = nullptr;
    float *y = nullptr;
    int out_dim = 0;
    int n_blocks = 0;
    bool use_wl = false;   // 尺寸门：大形状走字加载体

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolVQ2Wl() {
        const int p = default_parallelism_vq2_wl();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx_ = 1; idx_ < p; ++idx_) {
            workers.emplace_back([this, idx_] { worker_main(idx_); });
        }
    }

    ~RowPoolVQ2Wl() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
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

    void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        if (use_wl) run_range_wl(cb, idx_base, n_blocks, x, y, begin, end);
        else run_range_byte(cb, idx_base, n_blocks, x, y, begin, end);
    }

    void run(const uint8_t *w, const float *x_, float *y_, int out_dim_, int in_dim_,
             bool use_wl_) {
        load_codebook(w, cb);
        idx_base = w + kCodebookBytes;
        x = x_;
        y = y_;
        out_dim = out_dim_;
        n_blocks = in_dim_ / kBlockDim;
        use_wl = use_wl_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
    }
};

RowPoolVQ2Wl &pool_vq2_wl() {
    static RowPoolVQ2Wl p;
    return p;
}

// 粒度阈值（按元素数，与上一层同款）
constexpr std::size_t kMinParallelElems = 262144;

// 字加载体的尺寸门：索引区字节数 ≥ 8MB 才用字加载体（见文件头实测）
constexpr std::size_t kWordLoadMinIdxBytes = 8u << 20;

} // namespace

// ========================================================================
// matvec_vq2_neon_mr_mt_wl() — VQ2 matvec：4 行并行 + 多线程 + 字加载入口
// ========================================================================
void matvec_vq2_neon_mr_mt_wl(const uint8_t *w, const float *x, float *y,
                              int out_dim, int in_dim) {
    const int n_blocks = in_dim / kBlockDim;
    // 尺寸门：索引区字节数（= out_dim × n_blocks）≥ 8MB 才值得字加载
    const bool use_wl = static_cast<std::size_t>(out_dim) * n_blocks >=
                        kWordLoadMinIdxBytes;
    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElems ||
        std::thread::hardware_concurrency() <= 1) {
        float cb[kCodebookEntries][kBlockDim];
        load_codebook(w, cb);
        if (use_wl) run_range_wl(cb, w + kCodebookBytes, n_blocks, x, y, 0, out_dim);
        else run_range_byte(cb, w + kCodebookBytes, n_blocks, x, y, 0, out_dim);
        return;
    }
    pool_vq2_wl().run(w, x, y, out_dim, in_dim, use_wl);
}

// 自注册进 dispatch：matvec_vq2 的 "neon_mr_mt_wl" 实现
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon_mr_mt_wl, "neon_mr_mt_wl");

} // namespace tinyqwen

#endif // __aarch64__
