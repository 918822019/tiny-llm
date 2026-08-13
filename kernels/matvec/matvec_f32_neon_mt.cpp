// 矩阵乘向量：y = W @ x —— NEON + 多线程版。
//
// 在 neon 变体（逐行 NEON FMA 点积）之上叠加"多核并行读权重"：decode 是
// 纯权重带宽瓶颈（每 token 约 2GB 权重流量），单个 P 核只能跑出 ~72 GB/s；
// 把 out_dim 行切给多个常驻 worker 线程并行流式读取，有效带宽能逼近整机
// 内存带宽。每行的数值算法与 neon 完全一致（float 累加、同样的尾段处理），
// 只是行被分到了不同核上——所以对齐门禁与 neon 相同。
//
// 设计要点：
//   1. **常驻线程池**：decode 每个 token 要调 ~170 次 matvec，每次现起线程
//      的创建开销会比小矩阵本身还贵。worker 启动后自旋待命，由 master 用
//      原子代数（job_gen）release-store 唤醒——全程无 OS 锁、无系统调用；
//   2. **fork-join**：master 写任务参数 → release 递增 job_gen → 各 worker
//      acquire 到后代号算自己的行块 → done_gen.fetch_add 归位 → master
//      自己也算第 0 块，然后自旋等 done_gen 达标后返回；
//   3. **粒度阈值**：权重不足 1MB 的小矩阵（如 k/v_proj）直接单线程内联，
//      同步开销会吃掉并行收益；
//   4. **线程数**：默认 = 物理 P 核数 + 1（含 master 自己；实测再多带 1 个
//      E 核还能多挤一点带宽，更多就开始互踩，见 optimization_log 的 neon_mt
//      条目）；环境变量 TINYQWEN_MT_THREADS 可覆盖（调优/实验用）；非 Apple
//      平台无大小核信息，退回 hardware_concurrency。
//
// 选用：--matvec-impl neon_mt（仅 aarch64 构建注册；其他平台编译为空）。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
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
    // ---- 行点积：与 matvec_f32_neon 逐位一致（4 累加器 + FMA + 标量尾段）----
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

    // 自旋等待原子计数器到达目标值。先 YIELD 指令自旋，久了让出 CPU，
    // 避免 worker 在小间隙里空烧调度器。
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

    // 总并行度（含 master）。优先级：TINYQWEN_MT_THREADS > Apple P 核数 + 1 >
    // hardware_concurrency。上限 16，防止极端配置。
    // Apple 上取 P 核数 + 1：本机（M5 Pro，5P+10E）实测 5 线程 10.9ms/tok、
    // 6 线程 10.2、8 线程起反而抖动变差——带宽瓶颈下多 1 个 E 核能再挤一点
    // 带宽，再多就开始互相踩。其他平台没有大小核信息，退回全部核心。
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
      if (p <= 1) p = 2; // 只有 1 核时多线程无意义，交给调用侧的内联路径兜底
      return p > 16 ? 16 : p;
    }

    // 常驻行切分线程池。Meyers singleton：第一次用到才创建，进程退出时
    // 析构（通知 shutdown + join）。只有 master（调用方线程）驱动任务。
    struct RowPool {
      // 任务参数：仅 master 在发布新任务前写；worker 只在 acquire 到新的
      // job_gen 之后读，可见性由 release/acquire 保证。
      const float *w = nullptr;
      const float *x = nullptr;
      float *y = nullptr;
      int out_dim = 0;
      int in_dim = 0;

      std::atomic<std::uint64_t> job_gen{0};  // master 发布任务时递增
      std::atomic<std::uint64_t> done_gen{0}; // 每个 worker 完成一块后递增
      std::atomic<bool> shutdown{false};
      std::vector<std::thread> workers;
      std::uint64_t job_counter = 0;    // 仅 master 访问
      std::uint64_t expected_done = 0;  // 仅 master 访问

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

      // worker 循环：等任务 -> 算自己那块行 -> 归位。idx 从 1 起（0 是 master）。
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

      // 第 idx 块行区间上的逐行点积。行数不均时前 rem 块多分 1 行；
      // 行比并行度还少时，靠后的块区间为空，直接返回。
      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        for (int o = begin; o < end; ++o) {
          const float *row = w + static_cast<size_t>(o) * in_dim;
          y[o] = dot_row_neon(row, x, in_dim);
        }
      }

      // fork-join 入口（仅 master 调用）：发布任务 -> 自己算第 0 块 -> 等归位。
      void run(const float *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_;
        x = x_;
        y = y_;
        out_dim = out_dim_;
        in_dim = in_dim_;
        job_gen.store(++job_counter, std::memory_order_release); // 发布
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done); // join
      }
    };

    RowPool &pool() {
      static RowPool p; // 首次调用创建；线程安全（C++11 magic static）
      return p;
    }

    // 权重元素少于该值时不值得并行（同步开销 > 收益）。1MB = 262144 float：
    // q/o_proj（896×896，3.2MB）走并行；k/v_proj（128×896，0.45MB）走内联。
    constexpr std::size_t kMinParallelElems = 262144;
  } // namespace

  void matvec_f32_neon_mt(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<std::size_t>(out_dim) * in_dim;
    if (elems < kMinParallelElems || p.workers.empty()) {
      // 内联路径：与 neon 变体完全相同的单线程循环。
      for (int o = 0; o < out_dim; ++o) {
        const float *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row_neon(row, x, in_dim);
      }
      return;
    }
    p.run(w, x, y, out_dim, in_dim);
  }

  // 自注册进 dispatch：--matvec-impl neon_mt 即可选用（仅 aarch64 构建存在）。
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_mt, "neon_mt");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
