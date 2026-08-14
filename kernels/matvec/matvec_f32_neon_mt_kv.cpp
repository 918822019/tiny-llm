// 矩阵乘向量：y = W @ x —— NEON + 多线程 + k/v 成对融合版。
//
// 在 neon_mt（NEON 行点积 + 常驻线程池，行均分）之上只改一件事：
// **把共享同一输入向量的两个小 matvec 合成一次 fork-join**。
//
// ============================================================================
// 问题：k/v_proj 太小，够不着并行阈值
// ============================================================================
// Qwen2.5-0.5B 的 k_proj/v_proj 各是 128×896（0.45MB）。neon_mt 的粒度
// 阈值是 1MB——低于它的调用走单线程内联（同步开销会吃掉并行收益）。
// 于是每个 token 有 24 层 × 2 = 48 次 ~6µs 的内联小 matvec，合计约
// 0.3ms，全程单线程。
//
// 但 k 和 v 共享同一个输入向量（normed_），数学上完全可以一次算完：
// 把两个矩阵的行拼成一个 2×out_dim 行的 job，总量 0.9MB，行数 256——
// 一次 fork-join 摊到 6 个线程上，同步开销只付一次。这是"调用粒度的
// 合并"，不是新算法：每行的点积与 neon_mt 逐位一致，所以对齐门禁相同。
//
// ============================================================================
// 接入方式：dispatch 的成对入口（不影响其他 impl）
// ============================================================================
// dispatch 提供通用入口 matvec_pair_f32：当前 impl 注册过 pair 实现就用，
// 没注册就兜底为调两次 matvec_f32（数值不变）。本变体注册 pair 实现：
//
//   --matvec-impl neon_mt_kv：
//     q/o/gate/up/down/lm_head 走 matvec_f32_neon_mt_kv（与 neon_mt 同款）；
//     k/v 走 matvec_pair_f32_neon_mt_kv（合并 fork-join）。
//
// 其余 impl（ref/neon/neon_mt/...）不注册 pair，k/v 行为与从前完全一致。
// runtime 侧 qwen_model.cpp 把 k/v 两次调用改成了 matvec_pair_f32 一次。
//
// 粒度：pair 的并行阈值取两矩阵元素总和 ≥ 0.5MB——k+v（0.9MB）走并行；
// 更小的 pair 退回两段内联循环（等价于分开调）。
//
// 其余机制与 neon_mt 逐位一致：行点积（float 累加、尾段处理）、常驻线程池
// （原子代数唤醒、无 OS 锁）、单矩阵阈值（<1MB 内联）、默认线程数
// （P 核数 + 1，TINYQWEN_MT_THREADS 可覆盖）。行内算法没变，对齐门禁同。
//
// 选用：--matvec-impl neon_mt_kv（仅 aarch64 构建注册；其他平台编译为空）。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT / TINYQWEN_MATVEC_PAIR_VARIANT
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace tinyqwen {
  namespace {
    // ---- 行点积：与 neon_mt / neon 逐位一致（4 累加器 + FMA + 标量尾段）----
    inline float dot_row_neon(const float *row, const float *x, int n) {
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      const int n16 = n & ~15;
      for (; i < n16; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(row + i + 8), vld1q_f32(x + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
      }
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
      }
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }

    inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
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

    // 总并行度（含 master）：与 neon_mt 相同的优先级与取值。
    inline int default_parallelism() {
      if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
      }
      int p = 0;
#if defined(__APPLE__)
      int perf_cores = 0;
      std::size_t len = sizeof(perf_cores);
      if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &len, nullptr, 0) == 0) {
        p = perf_cores + 1;
      }
#endif
      if (p <= 0) p = static_cast<int>(std::thread::hardware_concurrency());
      if (p <= 1) p = 2;
      return p > 16 ? 16 : p;
    }

    // 常驻行切分线程池（neon_mt 的 RowPool + 成对任务支持）。
    struct RowPool {
      // 任务参数：仅 master 在发布新任务前写；worker 只在 acquire 到新的
      // job_gen 之后读。pair job 时 (w, y) 是第一个矩阵、(w2, y2) 第二个；
      // 单矩阵 job 时 w2 = nullptr。
      const float *w = nullptr;
      const float *w2 = nullptr;
      const float *x = nullptr;
      float *y = nullptr;
      float *y2 = nullptr;
      int out_dim = 0; // 单个矩阵的行数；pair job 的总行数 = 2*out_dim
      int in_dim = 0;
      bool pair = false;

      std::atomic<std::uint64_t> job_gen{0};
      std::atomic<std::uint64_t> done_gen{0};
      std::atomic<bool> shutdown{false};
      std::vector<std::thread> workers;
      std::uint64_t job_counter = 0;
      std::uint64_t expected_done = 0;

      RowPool() {
        const int p = default_parallelism();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
          workers.emplace_back([this, idx] { worker_main(idx); });
        }
      }

      ~RowPool() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) {
          t.join();
        }
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

      // 第 idx 块行区间。总行数 total = pair ? 2*out_dim : out_dim；
      // pair job 里行 r < out_dim 属于第一个矩阵，其余属于第二个。
      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int total = pair ? 2 * out_dim : out_dim;
        const int base = total / p;
        const int rem = total % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        for (int r = begin; r < end; ++r) {
          const int o = pair && r >= out_dim ? r - out_dim : r;
          const float *wm = pair && r >= out_dim ? w2 : w;
          float *ym = pair && r >= out_dim ? y2 : y;
          const float *row = wm + static_cast<size_t>(o) * in_dim;
          ym[o] = dot_row_neon(row, x, in_dim);
        }
      }

      // fork-join 发布（仅 master 调用）：写参数 -> 发布 -> 自己算第 0 块
      // -> 等归位。
      void publish_and_run() {
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
      }

      void run(const float *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_;
        w2 = nullptr;
        x = x_;
        y = y_;
        y2 = nullptr;
        out_dim = out_dim_;
        in_dim = in_dim_;
        pair = false;
        publish_and_run();
      }

      void run_pair(const float *w1_, const float *w2_, const float *x_, float *y1_,
                    float *y2_, int out_dim_, int in_dim_) {
        w = w1_;
        w2 = w2_;
        x = x_;
        y = y1_;
        y2 = y2_;
        out_dim = out_dim_;
        in_dim = in_dim_;
        pair = true;
        publish_and_run();
      }
    };

    RowPool &pool() {
      static RowPool p;
      return p;
    }

    // 单矩阵粒度阈值：与 neon_mt 相同（<1MB 内联）。
    constexpr std::size_t kMinParallelElems = 262144;
    // pair 粒度阈值：两矩阵元素总和 < 0.5MB 时不值得并行。k+v（0.9MB）过线。
    constexpr std::size_t kMinPairParallelElems = 131072;

    inline void matvec_inline(const float *w, const float *x, float *y, int out_dim,
                              int in_dim) {
      for (int o = 0; o < out_dim; ++o) {
        const float *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row_neon(row, x, in_dim);
      }
    }
  } // namespace

  // 单矩阵入口：与 neon_mt 相同的路径选择。
  void matvec_f32_neon_mt_kv(const float *w, const float *x, float *y, int out_dim,
                             int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (elems < kMinParallelElems || p.workers.empty()) {
      matvec_inline(w, x, y, out_dim, in_dim);
      return;
    }
    p.run(w, x, y, out_dim, in_dim);
  }

  // 成对入口：两矩阵行合并成一个 fork-join job。
  void matvec_pair_f32_neon_mt_kv(const float *w1, const float *w2, const float *x,
                                  float *y1, float *y2, int out_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (2 * elems < kMinPairParallelElems || p.workers.empty()) {
      matvec_inline(w1, x, y1, out_dim, in_dim);
      matvec_inline(w2, x, y2, out_dim, in_dim);
      return;
    }
    p.run_pair(w1, w2, x, y1, y2, out_dim, in_dim);
  }

  // 自注册：matvec 主入口 + pair 入口同名登记。仅 aarch64 构建存在。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_mt_kv, "neon_mt_kv");
  TINYQWEN_MATVEC_PAIR_VARIANT(matvec_pair_f32_neon_mt_kv, "neon_mt_kv");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
