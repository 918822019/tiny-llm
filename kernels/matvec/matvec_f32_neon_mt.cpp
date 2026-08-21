// ============================================================================
// matvec_f32_neon_mt.cpp — 矩阵乘向量：y = W @ x —— NEON + 多线程版
// ============================================================================
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
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"  // 辅助函数声明

// 平台开关：只有 aarch64 才编译下面的实现
#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h> // NEON intrinsic 声明

#include <atomic>     // std::atomic（线程同步用的原子计数器）
#include <cstddef>    // size_t
#include <cstdlib>    // std::getenv、std::strtol
#include <thread>     // std::thread（常驻 worker 线程）
#include <vector>     // std::vector（worker 线程容器）

#if defined(__APPLE__)
#include <sys/sysctl.h> // macOS 专用：查询 P 核数量
#endif

namespace tinyqwen {
  namespace {
    // ========================================================================
    // dot_row_neon() — 行点积：与 matvec_f32_neon 逐位一致
    // ========================================================================
    // 功能：计算 row[0..n-1] 与 x[0..n-1] 的点积（NEON 4 链 FMA）
    // 参数：row — fp32 权重行, x — fp32 输入向量, n — 向量长度
    // 返回值：点积结果（fp32）
    // 说明：4 个独立累加器 + FMA + 标量尾段，与 neon 单线程版完全相同
    inline float dot_row_neon(const float *row, const float *x, int n) {
      // 4 个独立 fp32 累加器初始化
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      // 主循环：一次迭代吃 16 个元素（4 链 × 4 lane）
      const int n16 = n & ~15; // n 向下取整到 16 的倍数
      for (; i < n16; i += 16) {
        // 4 条独立的 FMA 链，各处理 4 个连续元素
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));         // 链0: elem[i..i+3]
        acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4)); // 链1: elem[i+4..i+7]
        acc2 = vfmaq_f32(acc2, vld1q_f32(row + i + 8), vld1q_f32(x + i + 8)); // 链2: elem[i+8..i+11]
        acc3 = vfmaq_f32(acc3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12)); // 链3: elem[i+12..i+15]
      }
      // 向量尾段：还剩 4~15 个元素时按 4 个一批处理
      const int n4 = n & ~3; // n 向下取整到 4 的倍数
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
      }
      // 合并 4 个累加器：先两两相加再横向归约
      const float32x4_t sum01 = vaddq_f32(acc0, acc1); // 链0 + 链1
      const float32x4_t sum23 = vaddq_f32(acc2, acc3); // 链2 + 链3
      float total = vaddvq_f32(vaddq_f32(sum01, sum23)); // 4 lane 横向归约为标量
      // 标量尾段：还剩 0~3 个元素逐个补上
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }

    // ========================================================================
    // spin_until() — 自旋等待原子计数器到达目标值
    // ========================================================================
    // 功能：忙等直到原子变量 a 的值等于 target
    // 参数：a — 原子计数器, target — 期望值
    // 说明：前 256 次用 YIELD 指令自旋（让出流水线但不让出 CPU），之后
    //       调 std::this_thread::yield() 让出 CPU 时间片
    inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
      int spins = 0; // 自旋计数器
      while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
#if defined(__aarch64__)
          __builtin_arm_yield(); // ARM YIELD 指令：提示 CPU 这是自旋等待
#endif
        } else {
          std::this_thread::yield(); // 超过 256 次后让出 CPU 时间片
        }
      }
    }

    // ========================================================================
    // default_parallelism() — 计算默认并行度（含 master 线程）
    // ========================================================================
    // 功能：确定线程池的总线程数
    // 返回值：并行度（1~16）
    // 优先级：TINYQWEN_MT_THREADS > Apple P 核数 + 1 > hardware_concurrency
    // 说明：Apple 上取 P 核数 + 1：本机（M5 Pro，5P+10E）实测 5 线程 10.9ms/tok、
    //       6 线程 10.2、8 线程起反而抖动变差——带宽瓶颈下多 1 个 E 核能再挤一点
    //       带宽，再多就开始互相踩。
    inline int default_parallelism() {
      // 优先检查环境变量覆盖
      if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10); // 解析十进制整数
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v); // 上限 16
      }
      int p = 0;
#if defined(__APPLE__)
      // macOS 专用：通过 sysctl 查询性能核（P 核）的物理核心数
      int perf_cores = 0;
      std::size_t len = sizeof(perf_cores);
      if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &len, nullptr, 0) == 0) {
        p = perf_cores + 1; // P 核数 + 1 个额外线程
      }
#endif
      // 非 Apple 平台或 sysctl 失败时退回全部硬件线程数
      if (p <= 0) p = static_cast<int>(std::thread::hardware_concurrency());
      if (p <= 1) p = 2; // 至少 2 线程（单核时多线程无意义，靠内联路径兜底）
      return p > 16 ? 16 : p; // 上限 16
    }

    // ========================================================================
    // RowPool — 常驻行切分线程池
    // ========================================================================
    // Meyers singleton：第一次用到才创建，进程退出时析构（通知 shutdown + join）。
    // 只有 master（调用方线程）驱动任务。worker 自旋待命，全程无 OS 锁。
    struct RowPool {
      // ---- 任务参数：仅 master 在发布新任务前写；worker 只在 acquire 到新的
      //      job_gen 之后读，可见性由 release/acquire 保证 ----
      const float *w = nullptr;  // 权重矩阵
      const float *x = nullptr;  // 输入向量
      float *y = nullptr;        // 输出向量
      int out_dim = 0;           // 输出维度（行数）
      int in_dim = 0;            // 输入维度（列数）

      // ---- 同步原语 ----
      std::atomic<std::uint64_t> job_gen{0};  // master 发布任务时递增（release-store）
      std::atomic<std::uint64_t> done_gen{0}; // 每个 worker 完成一块后递增（release）
      std::atomic<bool> shutdown{false};      // 析构时置 true，通知 worker 退出
      std::vector<std::thread> workers;       // worker 线程容器
      std::uint64_t job_counter = 0;          // 仅 master 访问的任务计数器
      std::uint64_t expected_done = 0;        // 仅 master 访问的预期完成计数

      // 构造函数：创建 p-1 个 worker 线程（master 自己是第 0 号）
      RowPool() {
        const int p = default_parallelism();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
          workers.emplace_back([this, idx] { worker_main(idx); }); // 启动 worker
        }
      }

      // 析构函数：通知所有 worker 退出并 join
      ~RowPool() {
        shutdown.store(true, std::memory_order_release); // 设置关闭标志
        job_gen.fetch_add(1, std::memory_order_release); // 唤醒所有正在自旋等待的 worker
        for (auto &t : workers) {
          t.join(); // 等待每个 worker 线程结束
        }
      }

      // worker 主循环：等任务 → 算自己那块行 → 归位
      // 参数：idx — 线程编号（从 1 起，0 是 master）
      void worker_main(int idx) {
        std::uint64_t next_job = 1; // worker 期望的下一个任务代号
        for (;;) {
          spin_until(job_gen, next_job); // 自旋等待 master 发布新任务
          if (shutdown.load(std::memory_order_acquire)) return; // 收到关闭信号则退出
          do_chunk(idx); // 执行分配给自己的行块
          done_gen.fetch_add(1, std::memory_order_release); // 报告完成
          ++next_job; // 准备等下一个任务
        }
      }

      // 第 idx 块行区间上的逐行点积
      // 参数：idx — 线程编号
      // 说明：将 out_dim 行均分给 p 个线程，前 rem 块各多分 1 行；
      //       行比并行度还少时，靠后的块区间为空，直接返回。
      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1; // 总并行度
        const int base = out_dim / p;  // 每块的基础行数
        const int rem = out_dim % p;   // 余数（前 rem 块各多 1 行）
        // 线程 idx 的行区间 [begin, end)
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        // 遍历分配给本线程的每一行，做 NEON 点积
        for (int o = begin; o < end; ++o) {
          const float *row = w + static_cast<size_t>(o) * in_dim; // 定位行起点
          y[o] = dot_row_neon(row, x, in_dim); // 计算该行的输出分量
        }
      }

      // fork-join 入口（仅 master 调用）：发布任务 → 自己算第 0 块 → 等归位
      void run(const float *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_;         // 写入任务参数
        x = x_;
        y = y_;
        out_dim = out_dim_;
        in_dim = in_dim_;
        job_gen.store(++job_counter, std::memory_order_release); // 发布新任务
        do_chunk(0);    // master 自己算第 0 块
        expected_done += workers.size(); // 预期完成的 worker 数累加
        spin_until(done_gen, expected_done); // 等待所有 worker 完成（join）
      }
    };

    // Meyers singleton：首次调用创建；线程安全（C++11 magic static）
    RowPool &pool() {
      static RowPool p;
      return p;
    }

    // 权重元素少于该值时不值得并行（同步开销 > 收益）。1MB = 262144 float：
    // q/o_proj（896×896，3.2MB）走并行；k/v_proj（128×896，0.45MB）走内联。
    constexpr std::size_t kMinParallelElems = 262144;
  } // namespace

  // ========================================================================
  // matvec_f32_neon_mt() — 外部入口
  // ========================================================================
  // 功能：计算 y = W @ x（fp32，NEON + 多线程行切分）
  // 参数：
  //   w       — 权重矩阵，行主序 [out_dim, in_dim]
  //   x       — 输入向量（长度 in_dim）
  //   y       — 输出向量（长度 out_dim）
  //   out_dim — 输出维度
  //   in_dim  — 输入维度
  // 说明：权重总量低于阈值时走单线程内联路径，否则提交给线程池并行。
  void matvec_f32_neon_mt(const float *w, const float *x, float *y, int out_dim, int in_dim) {
    RowPool &p = pool(); // 获取全局线程池
    const std::size_t elems = static_cast<std::size_t>(out_dim) * in_dim; // 权重总元素数
    if (elems < kMinParallelElems || p.workers.empty()) {
      // 内联路径：与 neon 变体完全相同的单线程循环
      for (int o = 0; o < out_dim; ++o) {
        const float *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row_neon(row, x, in_dim);
      }
      return;
    }
    // 大矩阵走多线程路径
    p.run(w, x, y, out_dim, in_dim);
  }

  // 自注册进 dispatch：--matvec-impl neon_mt 即可选用（仅 aarch64 构建存在）
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_mt, "neon_mt");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
