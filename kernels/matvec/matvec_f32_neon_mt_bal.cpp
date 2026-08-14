// 矩阵乘向量：y = W @ x —— NEON + 多线程 + 自校准加权分块版。
//
// 在 neon_mt（NEON 行点积 + 常驻线程池）之上只改一件事：**行怎么分**。
//
// ============================================================================
// 问题：均分行数时，慢核成为尾巴
// ============================================================================
// neon_mt 的 do_chunk 把 out_dim 行**均分**给 p 个线程（±1 行）。但
// Apple Silicon 是大小核架构：本机 5 个超级核 + 10 个性能核（能效核），
// 默认并行度 = 5 + 1 = 6 时，那 1 个线程会被调度器放到慢核上。慢核的
// 单核流式带宽明显低于快核，却分到同样多的行——于是快核提前算完，
// 在 spin_until 里空转等它，并行段的时间被最慢的核拉长。
//
// 均分只有在"所有线程一样快"时才最优。线程速度不一样时，最优分法是
// **按吞吐比例分行**：快核多分、慢核少分，让所有线程同时完工。
//
// ============================================================================
// 方案：自校准加权分块（本变体唯一的新增机制）
// ============================================================================
// 不去探测"哪个线程在哪个核"（macOS 没有硬亲和，线程还会迁移），
// 而是直接**测量**每个线程的真实吞吐：
//
//   1. 每个 worker 算完自己的行块后，记录耗时（steady_clock）；
//   2. master 在 join 归位后读到所有耗时（done_gen 的 release/acquire
//      同时保证了数据的可见性），换算成 rows/ns，用 EMA 更新每线程的
//      速度估计（新样本权重 0.3：既跟得上调度变化，又不被单次噪声带偏）；
//   3. 下一个 job 发布前，master 按速度比例把 out_dim 行切成 p 段
//      （前缀和取整，误差 ≤1 行），写进 bounds[]。
//
// 热循环里没有任何原子操作（对比"共享计数器抢行"方案：小矩阵上原子
// 开销占比太高）；新增开销只有每 job 两次 steady_clock::now()（~20ns）
// 和 master 侧 O(p) 的分块计算——对 µs~ms 级的并行 job 可忽略。
//
// 首个并行 job 还没有测量数据，退回均分（与 neon_mt 相同）；decode 每
// token 有 ~145 个并行 matvec，EMA 在预fill 阶段就已收敛，bench 丢弃的
// 前 4 个预热 token 足够把权重喂稳。
//
// 其余一切与 neon_mt 逐位一致：行点积算法（float 累加、尾段处理）、
// 常驻线程池（原子代数唤醒、无 OS 锁）、粒度阈值（<1MB 内联单线程）、
// 默认线程数（P 核数 + 1，TINYQWEN_MT_THREADS 可覆盖）。所以对齐
// 门禁与 neon_mt 相同，只是行的归属不同（每行内部算法没变）。
//
// 选用：--matvec-impl neon_mt_bal（仅 aarch64 构建注册；其他平台编译为空）。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
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

    // 自旋等待原子计数器到达目标值（与 neon_mt 相同：先 YIELD 自旋，
    // 久了让出 CPU）。
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

    inline std::uint64_t now_ns() {
      return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count());
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

    // 常驻行切分线程池（在 neon_mt 的 RowPool 上加了速度自校准与加权分块）。
    struct RowPool {
      static constexpr int kMaxThreads = 16;

      // 任务参数：仅 master 在发布新任务前写；worker 只在 acquire 到新的
      // job_gen 之后读，可见性由 release/acquire 保证。
      const float *w = nullptr;
      const float *x = nullptr;
      float *y = nullptr;
      int out_dim = 0;
      int in_dim = 0;

      // 加权分块结果：p+1 个边界（行前缀和），bounds[i]..bounds[i+1) 是
      // 线程 i 的行区间。master 在发布前计算（O(p)），worker 只读。
      int bounds[kMaxThreads + 1] = {};

      std::atomic<std::uint64_t> job_gen{0};  // master 发布任务时递增
      std::atomic<std::uint64_t> done_gen{0}; // 每个 worker 完成一块后递增
      std::atomic<bool> shutdown{false};
      std::vector<std::thread> workers;
      std::uint64_t job_counter = 0;    // 仅 master 访问
      std::uint64_t expected_done = 0;  // 仅 master 访问

      // ---- 速度自校准状态 ----
      // chunk_ns[i]：线程 i 上一个 job 的块耗时（线程 i 写，master 在
      // join 后读——done_gen 的 release/acquire 建立 happens-before）。
      std::uint64_t chunk_ns[kMaxThreads] = {};
      // rows_done[i]：线程 i 上一个 job 分到的行数（master 前后都知道）。
      int rows_done[kMaxThreads] = {};
      // speed[i]：线程 i 的吞吐 EMA 估计（rows/ms，仅 master 读写）。
      double speed[kMaxThreads] = {};
      bool calibrated = false; // 是否已积累过至少一轮完整测量
      static constexpr double kEmaNew = 0.3; // 新样本权重

      RowPool() {
        const int p = default_parallelism();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
          workers.emplace_back([this, idx] { worker_main(idx); });
        }
      }

      ~RowPool() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release); // 唤醒所有 worker 退出
        for (auto &t : workers) {
          t.join();
        }
      }

      // worker 循环：等任务 -> 计时算自己那块行 -> 记耗时 -> 归位。
      void worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
          spin_until(job_gen, next_job);
          if (shutdown.load(std::memory_order_acquire)) return;
          const std::uint64_t t0 = now_ns();
          do_chunk(idx);
          chunk_ns[idx] = now_ns() - t0;
          done_gen.fetch_add(1, std::memory_order_release);
          ++next_job;
        }
      }

      // 线程 idx 的行区间上的逐行点积。区间来自 bounds[]（加权分块）。
      void do_chunk(int idx) const {
        const int begin = bounds[idx];
        const int end = bounds[idx + 1];
        for (int o = begin; o < end; ++o) {
          const float *row = w + static_cast<size_t>(o) * in_dim;
          y[o] = dot_row_neon(row, x, in_dim);
        }
      }

      // 按速度比例切行（仅 master 调用，发布前）。
      // 未校准时退化为均分（与 neon_mt 相同）；校准后按 speed 前缀和切：
      // 快线程多分、慢线程少分，目标是所有线程同时完工。
      void compute_bounds() {
        const int p = static_cast<int>(workers.size()) + 1;
        if (!calibrated) {
          // 均分（neon_mt 原逻辑）：前 rem 块各多 1 行。
          const int base = out_dim / p;
          const int rem = out_dim % p;
          int b = 0;
          for (int i = 0; i < p; ++i) {
            bounds[i] = b;
            b += base + (i < rem ? 1 : 0);
          }
          bounds[p] = out_dim;
          return;
        }
        double total = 0.0;
        for (int i = 0; i < p; ++i) total += speed[i];
        if (total <= 0.0) { // 防御：权重全零时退回均分
          calibrated = false;
          compute_bounds();
          return;
        }
        // 前缀和取整：bounds[i] = round(out_dim * cum_weight_i)。
        // 单调不减、首 0 尾 out_dim，每段行数误差 ≤1。
        bounds[0] = 0;
        double cum = 0.0;
        for (int i = 1; i < p; ++i) {
          cum += speed[i - 1];
          bounds[i] = static_cast<int>(out_dim * cum / total + 0.5);
        }
        bounds[p] = out_dim;
        for (int i = 0; i < p; ++i) rows_done[i] = bounds[i + 1] - bounds[i];
      }

      // join 之后用本轮实测更新速度 EMA（仅 master 调用）。
      void update_speeds() {
        const int p = static_cast<int>(workers.size()) + 1;
        // 耗时过短（<0.5µs）或空块的测量信噪比太低，不采样。
        auto valid = [&](int i) { return rows_done[i] > 0 && chunk_ns[i] > 500; };
        if (!calibrated) {
          // 首轮：有测量的直接用作初值；没测量的（空块/太短）用本轮
          // 最小测量值兜底（保守：宁可少分，不制造新尾巴）。
          double min_meas = 0.0;
          bool any = false;
          for (int i = 0; i < p; ++i) {
            if (valid(i)) {
              const double meas = static_cast<double>(rows_done[i]) * 1e6 /
                                  static_cast<double>(chunk_ns[i]); // rows/ms
              speed[i] = meas;
              if (!any || meas < min_meas) min_meas = meas;
              any = true;
            }
          }
          if (!any) return; // 全员不可测：维持均分
          for (int i = 0; i < p; ++i) {
            if (!valid(i)) speed[i] = min_meas;
          }
          calibrated = true;
          return;
        }
        for (int i = 0; i < p; ++i) {
          if (!valid(i)) continue;
          const double meas = static_cast<double>(rows_done[i]) * 1e6 /
                              static_cast<double>(chunk_ns[i]); // rows/ms
          speed[i] = (1.0 - kEmaNew) * speed[i] + kEmaNew * meas;
        }
      }

      // fork-join 入口（仅 master 调用）：切块 -> 发布 -> 自己算第 0 块
      // -> 等归位 -> 更新速度估计。
      void run(const float *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_;
        x = x_;
        y = y_;
        out_dim = out_dim_;
        in_dim = in_dim_;
        compute_bounds();
        // 均分路径（未校准）也要记行数，供首轮校准使用。
        if (!calibrated) {
          const int p = static_cast<int>(workers.size()) + 1;
          for (int i = 0; i < p; ++i) rows_done[i] = bounds[i + 1] - bounds[i];
        }
        const std::uint64_t t0 = now_ns();
        job_gen.store(++job_counter, std::memory_order_release); // 发布
        do_chunk(0);
        chunk_ns[0] = now_ns() - t0;
        expected_done += workers.size();
        spin_until(done_gen, expected_done); // join
        update_speeds();
      }
    };

    RowPool &pool() {
      static RowPool p; // 首次调用创建；线程安全（C++11 magic static）
      return p;
    }

    // 与 neon_mt 相同的粒度阈值：权重不足 1MB 直接单线程内联。
    constexpr std::size_t kMinParallelElems = 262144;
  } // namespace

  void matvec_f32_neon_mt_bal(const float *w, const float *x, float *y, int out_dim,
                              int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (elems < kMinParallelElems || p.workers.empty()) {
      for (int o = 0; o < out_dim; ++o) {
        const float *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row_neon(row, x, in_dim);
      }
      return;
    }
    p.run(w, x, y, out_dim, in_dim);
  }

  // 自注册进 dispatch：--matvec-impl neon_mt_bal 即可选用（仅 aarch64 构建存在）。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_mt_bal, "neon_mt_bal");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
