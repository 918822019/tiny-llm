// ============================================================================
// matvec_vq2_neon_mr_mt.cpp — VQ2 matvec：neon_mr + 常驻线程池行切分
// ============================================================================
// matvec_vq2_neon_mr 的归因阶梯下一层。布局与反量化完全一致；本文件在
// 4 行并行（独立 FMA 链 + 共享 x4 加载）之上只加一个技术：**多核行切分**，
// 线程池结构与 i4_neon_mt / f16 neon_mt_kv_nt 的 RowPool 相同
// （spin barrier、常驻 worker、TINYQWEN_MT_THREADS 覆盖、默认硬件核数上限 16）。
//
// 数值语义：每行的累加链相互独立，行区间如何划分不改变任何一行内的
// 浮点运算顺序——多线程结果与 neon_mr 单线程逐位等价（仅码本转换由主线程
// 统一做一次，驻留池内供 worker 共读）。
//
// 粒度阈值：元素数 < 262144（与 i4 族同款常量）走单线程，省同步开销。
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

// ========================================================================
// run_range() — 行区间 [begin, end) 的 4 行并行点积（与 neon_mr 主体同款）
// ========================================================================
// 每行累加链独立，区间划分不影响数值； batching 只在区间内做。
inline void run_range(const float cb[kCodebookEntries][kBlockDim],
                      const uint8_t *idx, int n_blocks,
                      const float *x, float *y, int begin, int end) {
    int o = begin;
    // 主体：每次 4 行 —— 4 条独立 FMA 链 + 共享 x4 加载
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
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim); // 4 行共享
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
    // 尾部（区间内不足 4 行）：单行路径
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

// 码本 [K, d] fp16 → fp32（一次性，4KB，跨行/跨线程复用）
inline void load_codebook(const uint8_t *w, float cb[kCodebookEntries][kBlockDim]) {
    for (int k = 0; k < kCodebookEntries; ++k) {
        for (int j = 0; j < kBlockDim; ++j) {
            uint16_t h;
            std::memcpy(&h, w + (k * kBlockDim + j) * 2, 2);
            cb[k][j] = half_to_float(h);
        }
    }
}

// 自旋等待原子计数器到达目标值（同 i4_neon_mt）
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

// 并行度：TINYQWEN_MT_THREADS 覆盖；默认 = 硬件核数（上限 16）
inline int default_parallelism_vq2() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolVQ2 — 常驻行切分线程池（结构同 RowPoolI4）
// ========================================================================
struct RowPoolVQ2 {
    // 调用上下文（主线程写于 fork 前，worker 在 barrier 后读）
    float cb[kCodebookEntries][kBlockDim]; // 主线程统一转换的码本（共读）
    const uint8_t *idx_base = nullptr;     // 索引区起点
    const float *x = nullptr;
    float *y = nullptr;
    int out_dim = 0;
    int n_blocks = 0;

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolVQ2() {
        const int p = default_parallelism_vq2();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx_ = 1; idx_ < p; ++idx_) {
            workers.emplace_back([this, idx_] { worker_main(idx_); });
        }
    }

    ~RowPoolVQ2() {
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

    // 第 idx 块行区间
    void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        run_range(cb, idx_base, n_blocks, x, y, begin, end);
    }

    // fork-join 入口：主线程先转码本，再派发
    void run(const uint8_t *w, const float *x_, float *y_, int out_dim_, int in_dim_) {
        load_codebook(w, cb);
        idx_base = w + kCodebookBytes;
        x = x_;
        y = y_;
        out_dim = out_dim_;
        n_blocks = in_dim_ / kBlockDim;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
    }
};

// Meyers singleton
RowPoolVQ2 &pool_vq2() {
    static RowPoolVQ2 p;
    return p;
}

// 粒度阈值（按元素数）：小矩阵单线程，省同步开销（与 i4 族同款）
constexpr std::size_t kMinParallelElems = 262144;

} // namespace

// ========================================================================
// matvec_vq2_neon_mr_mt() — VQ2 matvec：4 行并行 + 多线程入口
// ========================================================================
void matvec_vq2_neon_mr_mt(const uint8_t *w, const float *x, float *y,
                           int out_dim, int in_dim) {
    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElems ||
        std::thread::hardware_concurrency() <= 1) {
        // 单线程路径（小形状）：码本栈上转换，区间 = 全量
        float cb[kCodebookEntries][kBlockDim];
        load_codebook(w, cb);
        run_range(cb, w + kCodebookBytes, in_dim / kBlockDim, x, y, 0, out_dim);
        return;
    }
    pool_vq2().run(w, x, y, out_dim, in_dim);
}

// 自注册进 dispatch：matvec_vq2 的 "neon_mr_mt" 实现
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon_mr_mt, "neon_mr_mt");

} // namespace tinyqwen

#endif // __aarch64__
