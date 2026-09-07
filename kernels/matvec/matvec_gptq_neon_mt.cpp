// ============================================================================
// matvec_gptq_neon_mt.cpp — GPTQ-INT4 matvec NEON + 多线程（归因阶梯 L3）
// ============================================================================
// 在 neon 变体之上加常驻线程池 + 原子计数器动态领取 o_block。照 matvec_i4_sdot4
// 的 RowPoolSdot3 样板：动态调度让快核（P 簇）多领、慢核（E 簇）少领，总时长
// ≈ 总工作量 ÷ 聚合吞吐，而不是被最慢的一份静态切分拖住。
//
// 并行单位是 **o_block（64 列）**而非单行：GPTQ 的一个块处理要遍历全部
// in_dim/8 个字，是足够大的工作单元，粒度天然合适；按行切会让每次领取的
// 工作量太小、原子开销占比上升。
// ============================================================================

#include "dispatch.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include "matvec_gptq_neon_common.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

inline void spin_until_gptq(const std::atomic<std::uint64_t> &a,
                            std::uint64_t target) {
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

// 默认并行度：硬件并发数（上限 16），TINYQWEN_MT_THREADS 可覆盖。
// 敢用满全部核是因为动态调度消除了静态切分的慢核拖尾问题。
inline int default_parallelism_gptq() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// BlockPoolGptq — 常驻线程池 + 原子块计数器动态领取 o_block
// ========================================================================
struct BlockPoolGptq {
    GptqBlockView bv;
    const float *x = nullptr;
    float *y = nullptr;
    const float *sx8 = nullptr;
    const int *g_of_c8 = nullptr;
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 128;
    bool uniform = true;
    int n_blocks = 0;
    std::atomic<int> next_block{0};

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    BlockPoolGptq() {
        const int p = default_parallelism_gptq();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ~BlockPoolGptq() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void worker_main(int) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_gptq(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_work();
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    // 动态领取：每次 fetch_add 一个块号，处理完再领，直到领完。
    void do_work() {
        for (;;) {
            const int b = next_block.fetch_add(1, std::memory_order_relaxed);
            if (b >= n_blocks) break;
            const int ob = b * kGptqOBlock;
            const int blen =
                (out_dim - ob < kGptqOBlock) ? out_dim - ob : kGptqOBlock;
            if (uniform) {
                gptq_neon_block(bv, x, y, sx8, g_of_c8, out_dim, in_dim, ob, blen);
            } else {
                gptq_neon_slow_block(bv, x, y, out_dim, in_dim, group_size, ob, blen);
            }
        }
    }

    void run(const GptqBlockView &bv_, const float *x_, float *y_,
             const float *sx8_, const int *g_of_c8_, int out_dim_, int in_dim_,
             int group_size_, bool uniform_) {
        bv = bv_; x = x_; y = y_;
        sx8 = sx8_; g_of_c8 = g_of_c8_;
        out_dim = out_dim_; in_dim = in_dim_;
        group_size = group_size_; uniform = uniform_;
        n_blocks = (out_dim + kGptqOBlock - 1) / kGptqOBlock;
        next_block.store(0, std::memory_order_relaxed);
        // release：上面的字段写入对看到 job_gen 的 worker 全部可见
        job_gen.store(++job_counter, std::memory_order_release);
        do_work();  // master 同样参与领取
        expected_done += workers.size();
        spin_until_gptq(done_gen, expected_done);
    }
};

BlockPoolGptq &pool_gptq() {
    static BlockPoolGptq p;
    return p;
}

// 粒度阈值：总元素数低于此值走单线程，避免小矩阵上同步开销吃掉并行收益。
// 与 matvec_i4_sdot4 的 kMinParallelElemsSdot3 同值，便于跨 dtype 对照归因。
constexpr std::size_t kMinParallelElemsGptq = 262144;

void matvec_gptq_neon_mt(const uint8_t *w, const float *x, float *y, int out_dim,
                         int in_dim, int group_size) {
    if (in_dim % kGptqPackInts != 0 || in_dim % group_size != 0) {
        matvec_gptq_ref(w, x, y, out_dim, in_dim, group_size);
        return;
    }

    const GptqBlockView bv = parse_gptq_block(w, out_dim, in_dim, group_size);
    const int n_c8 = in_dim / kGptqPackInts;

    std::vector<float> sx8(static_cast<size_t>(n_c8));
    std::vector<int> g_of_c8(static_cast<size_t>(n_c8));
    gptq_precompute(x, in_dim, group_size, bv, sx8.data(), g_of_c8.data());

    const bool uniform = gptq_g_idx_uniform(bv.g_idx, in_dim);

    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElemsGptq ||
        std::thread::hardware_concurrency() <= 1) {
        // 小矩阵单线程：结构与 matvec_gptq_neon 相同
        for (int ob = 0; ob < out_dim; ob += kGptqOBlock) {
            const int blen =
                (out_dim - ob < kGptqOBlock) ? out_dim - ob : kGptqOBlock;
            if (uniform) {
                gptq_neon_block(bv, x, y, sx8.data(), g_of_c8.data(), out_dim,
                                in_dim, ob, blen);
            } else {
                gptq_neon_slow_block(bv, x, y, out_dim, in_dim, group_size, ob,
                                     blen);
            }
        }
        return;
    }

    pool_gptq().run(bv, x, y, sx8.data(), g_of_c8.data(), out_dim, in_dim,
                    group_size, uniform);
}

} // namespace

TINYQWEN_MATVEC_GPTQ_VARIANT(matvec_gptq_neon_mt, "neon_mt");

} // namespace tinyqwen

#else // 非 ARM：兜底到 ref

namespace tinyqwen {
namespace {

void matvec_gptq_neon_mt(const uint8_t *w, const float *x, float *y, int out_dim,
                         int in_dim, int group_size) {
    matvec_gptq_ref(w, x, y, out_dim, in_dim, group_size);
}

} // namespace

TINYQWEN_MATVEC_GPTQ_VARIANT(matvec_gptq_neon_mt, "neon_mt");

} // namespace tinyqwen

#endif
