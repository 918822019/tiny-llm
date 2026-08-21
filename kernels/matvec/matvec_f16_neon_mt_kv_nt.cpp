// ============================================================================
// matvec_f16_neon_mt_kv_nt.cpp — 矩阵乘向量（f16 权重）：y = W @ x
//                  NEON + 多线程 + k/v 融合 + LDNP 流式加载版
// ============================================================================
// 本文件是 f16 族的"满栈变体"（对应 f32 族的 matvec_f32_neon_mt_kv_nt），
// 把 f32 阶梯顶层的全部优化移植到半精度权重上。
//
// 核心特征：
//   - 权重是 IEEE binary16（每 token 的权重流量直接减半），x/y 保持 fp32
//     （weight-only：只省搬运，计算精度不降）；
//   - 4 链 FMA 累加：每链用 vld1q_f16 一次载 8 个 half，vcvt 转成两半
//     float32x4 后 FMA——累加仍在 fp32，不引入 fp16 累加的精度风险；
//   - 常驻线程池行切分、粒度阈值、k/v 成对融合：与 f32 版相同；
//   - LDNP 流式加载：权重按 ldnp q 对（32B = 16 half）非时间加载，
//     行 16B 对齐检查 + 非对齐兜底。真实模型里行步长 = in_dim×2，
//     896/4864/128 都是 8 的倍数 → 逐行 16B 对齐成立。
//
// 与 f32 版的逐项对应（行内数学语义不变，只是权重加载变窄）：
//   - f32 用 vld1q_f32 一次载 4 个 float → f16 用 vld1q_f16 一次载 8 个 half
//   - f32 主循环每次吃 16 float → f16 主循环每次吃 32 half（同样 4 条累加链）
//   - f32 LDNP 一次读 32B = 8 float → f16 LDNP 一次读 32B = 16 half
//
// f16 特有的数值约定：与 matvec_f16_ref（double 累加）的差只来自
// float 累加顺序，量级与 f32 族相同，门禁容差沿用 5e-3。
//
// 注册进 f16 注册表，实现名与 f32 阶梯顶层**同名** "neon_mt_kv_nt"——
// 选谁由模型文件的 dtype 决定（main 按 dtype 查表，见 dispatch.h 注释）。
// 仅 aarch64 构建注册；其他平台 f16 模型自动用 matvec_f16_ref 兜底。
//
// 提供三个入口：
//   1. matvec_f16_neon_mt_kv_nt      — 单矩阵 matvec
//   2. matvec_pair_f16_neon_mt_kv_nt — 两矩阵共享输入向量的成对 matvec（k/v 融合）
//   3. matvec_qkv_f16_neon_mt_kv_nt  — Q/K/V 三矩阵融合（QKV 投影一步完成）
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT / _F16_PAIR_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

// 平台开关：只有 aarch64 才编译下面的 NEON 实现；其他平台整个文件为空翻译单元
#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h> // ARM NEON intrinsic 声明（编译器自带）

#include <atomic>     // std::atomic（线程同步用的原子计数器）
#include <cstddef>    // size_t、uintptr_t
#include <cstdint>    // uint16_t、uint64_t
#include <cstdlib>    // std::getenv、std::strtol
#include <thread>     // std::thread（常驻 worker 线程）
#include <vector>     // std::vector（worker 线程容器）

#if defined(__APPLE__)
#include <sys/sysctl.h> // macOS 专用：查询 P 核数量（hw.perflevel0.physicalcpu）
#endif

namespace tinyqwen {
  namespace {
    // ========================================================================
    // dot_row_f16() — 普通行点积（f16 权重），对齐兜底路径
    // ========================================================================
    // 功能：计算 row[0..n-1] 与 x[0..n-1] 的点积，权重为 float16_t
    // 参数：
    //   row — f16 权重行的起始地址
    //   x   — fp32 输入向量
    //   n   — 向量长度（in_dim）
    // 返回值：点积结果（fp32）
    // 说明：使用普通 vld1q_f16 加载权重（不走 LDNP），当行地址不满足
    //       16B 对齐时作为兜底路径。数值结果与 LDNP 版完全一致。
    inline float dot_row_f16(const float16_t *row, const float *x, int n) {
      // 4 个独立 fp32 累加器：互不依赖，让 CPU 乱序执行重叠 FMA 延迟
      float32x4_t acc0 = vdupq_n_f32(0.0f); // 全零初始化
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      // ---- 主循环：一次迭代吃 32 个 half（4 链 × 每链 vld1q_f16 载 8 个）----
      // n32 = n 向下取整到 32 的倍数（抹掉低 5 位），保证 i+31 不越界
      const int n32 = n & ~31;
      for (; i < n32; i += 32) {
        // vld1q_f16：从内存连续加载 8 个 float16 到 128 位 NEON 寄存器
        float16x8_t h0 = vld1q_f16(row + i);      // half[i..i+7]
        float16x8_t h1 = vld1q_f16(row + i + 8);  // half[i+8..i+15]
        float16x8_t h2 = vld1q_f16(row + i + 16); // half[i+16..i+23]
        float16x8_t h3 = vld1q_f16(row + i + 24); // half[i+24..i+31]
        // vget_low_f16(h0)：取 h0 的低 4 个 half（64 位）
        // vcvt_f32_f16(...)：将 4 个 float16 转为 4 个 float32（float32x4_t）
        // vfmaq_f32(acc, a, b)：acc += a * b（逐 lane FMA，fp32 精度累加）
        // vld1q_f32(x+i)：加载 4 个 fp32 输入
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + i));
        // vcvt_high_f32_f16(h0)：取 h0 的高 4 个 half 并转为 float32x4_t
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h0), vld1q_f32(x + i + 4));
        // 以下三条链同理，各处理 8 个 half
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + i + 8));
        acc1 = vfmaq_f32(acc1, vcvt_high_f32_f16(h1), vld1q_f32(x + i + 12));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(h2)), vld1q_f32(x + i + 16));
        acc2 = vfmaq_f32(acc2, vcvt_high_f32_f16(h2), vld1q_f32(x + i + 20));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_low_f16(h3)), vld1q_f32(x + i + 24));
        acc3 = vfmaq_f32(acc3, vcvt_high_f32_f16(h3), vld1q_f32(x + i + 28));
      }
      // ---- 尾段：8 个一批（还剩 8~31 个元素时）----
      const int n8 = n & ~7; // n 向下取整到 8 的倍数
      for (; i < n8; i += 8) {
        // 一条链吃 8 个 half（高低各 4 个转 fp32 后 FMA）
        float16x8_t h = vld1q_f16(row + i);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
      }
      // ---- 尾段：4 个一批（还剩 4~7 个元素时）----
      const int n4 = n & ~3; // n 向下取整到 4 的倍数
      for (; i < n4; i += 4) {
        // vld1_f16：加载 4 个 half（64 位），vcvt_f32_f16 转成 float32x4_t
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16(row + i)), vld1q_f32(x + i));
      }
      // ---- 合并 4 个累加器 ----
      // vaddq_f32：逐 lane 相加，sum01[lane] = acc0[lane] + acc1[lane]
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      // vaddvq_f32：横向归约，把 4 个 lane 加成一个标量
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      // ---- 标量尾段：还剩 0~3 个元素，逐个补上 ----
      for (; i < n; ++i) {
        // static_cast<float>(row[i])：将 float16_t 隐式提升为 float
        total += static_cast<float>(row[i]) * x[i];
      }
      return total;
    }

    // ========================================================================
    // ldnp_pair_f16() — LDNP 流式加载一对 q 寄存器（32B = 16 half）
    // ========================================================================
    // 功能：用 LDNP（Load Non-temporal Pair）指令从地址 p 加载 32 字节到
    //       两个 128 位 NEON 寄存器 lo 和 hi，带非时间提示（不污染 cache）。
    // 参数：
    //   p  — 源地址（必须 16B 对齐，否则 fault）
    //   lo — 输出：低 16 字节（8 个 half）
    //   hi — 输出：高 16 字节（8 个 half）
    // 说明：LDNP 比两条普通 LDP 少一条指令，且给硬件"不必缓存"的提示。
    //       内联汇编中 %q 让编译器分配 q 寄存器编号；=&w 是 early-clobber
    //       约束，防止输出寄存器与输入指针寄存器冲突。
    inline void ldnp_pair_f16(const float16_t *p, float16x8_t &lo, float16x8_t &hi) {
      __asm__("ldnp %q[lo], %q[hi], [%[p]]"
              : [lo] "=&w"(lo), [hi] "=&w"(hi)
              : [p] "r"(p));
    }

    // ========================================================================
    // dot_row_f16_nt() — 流式行点积：权重走 LDNP，x 保持普通加载
    // ========================================================================
    // 功能：与 dot_row_f16 计算相同的点积，但权重加载改用 LDNP 指令。
    // 参数：同 dot_row_f16
    // 返回值：点积结果（fp32）
    // 说明：x 向量是热数据（每行都重用），要保持留在 cache 里，所以 x 仍
    //       用普通 vld1q_f32 加载；只有权重（纯流式访问）适用非时间提示。
    //       主循环一次 32 个 half = 2 条 LDNP（各载 16 half = 2 个 q 寄存器）。
    inline float dot_row_f16_nt(const float16_t *row, const float *x, int n) {
      // 4 个独立 fp32 累加器
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      // n32 = n 向下取整到 32 的倍数
      const int n32 = n & ~31;
      for (; i < n32; i += 32) {
        // 用 LDNP 加载 4 组各 8 个 half（共 32 个 half = 64 字节权重）
        float16x8_t h0, h1, h2, h3;
        ldnp_pair_f16(row + i, h0, h1);      // LDNP 加载 half[i..i+15]（32B）
        ldnp_pair_f16(row + i + 16, h2, h3); // LDNP 加载 half[i+16..i+31]（32B）
        // 以下 FMA 逻辑与 dot_row_f16 完全一致：每条链的高低 4 half 分别转 fp32 后 FMA
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h0), vld1q_f32(x + i + 4));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + i + 8));
        acc1 = vfmaq_f32(acc1, vcvt_high_f32_f16(h1), vld1q_f32(x + i + 12));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(h2)), vld1q_f32(x + i + 16));
        acc2 = vfmaq_f32(acc2, vcvt_high_f32_f16(h2), vld1q_f32(x + i + 20));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_low_f16(h3)), vld1q_f32(x + i + 24));
        acc3 = vfmaq_f32(acc3, vcvt_high_f32_f16(h3), vld1q_f32(x + i + 28));
      }
      // 尾段：8 个一批（退回普通 vld1q_f16 加载）
      const int n8 = n & ~7;
      for (; i < n8; i += 8) {
        float16x8_t h = vld1q_f16(row + i);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
      }
      // 尾段：4 个一批
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16(row + i)), vld1q_f32(x + i));
      }
      // 合并 4 个累加器 + 横向归约
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      // 标量尾段
      for (; i < n; ++i) {
        total += static_cast<float>(row[i]) * x[i];
      }
      return total;
    }

    // ========================================================================
    // dot_row() — 行点积入口：根据对齐情况选择 LDNP 或普通路径
    // ========================================================================
    // 功能：检查行地址是否 16B 对齐，对齐则走 LDNP 流式路径，否则兜底普通路径。
    // 参数：同 dot_row_f16
    // 返回值：点积结果（fp32）
    // 说明：LDNP 要求地址 16B 对齐，不对齐会触发 fault。真实模型里所有权重的
    //       行起点都满足对齐（tensor 基址 64B 对齐 × in_dim 是偶数），但 kernel
    //       不假设调用方，逐行检查更安全。
    inline float dot_row(const float16_t *row, const float *x, int n) {
      // 检查行地址的低 4 位是否为 0（即 16B 对齐）
      if ((reinterpret_cast<std::uintptr_t>(row) & 15) != 0) {
        // 未对齐：退回普通 vld1q_f16 加载路径
        return dot_row_f16(row, x, n);
      }
      // 对齐：走 LDNP 流式加载路径
      return dot_row_f16_nt(row, x, n);
    }

    // ========================================================================
    // spin_until() — 自旋等待原子计数器到达目标值
    // ========================================================================
    // 功能：忙等直到原子变量 a 的值等于 target。
    // 参数：
    //   a      — 要监控的原子计数器
    //   target — 期望的目标值
    // 说明：前 256 次自旋用 YIELD 指令（让出流水线但不让出 CPU），之后调
    //       std::this_thread::yield() 让出 CPU 时间片。避免 worker 在小间隙
    //       里空烧调度器。memory_order_acquire 确保读到 target 时能看到 master
    //       release-store 之前写入的任务参数。
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
    // 功能：确定线程池的总线程数。
    // 返回值：并行度（1~16）
    // 优先级：TINYQWEN_MT_THREADS 环境变量 > Apple P 核数 + 3 > hardware_concurrency
    // 说明：f16 版比 f32 版多 +3（而非 +1），因为 f16 把每 token 权重流量减半，
    //       单核更早撞到"总带宽还没吃满"而不是"单核份额到顶"，多几个 E 核能再挤
    //       带宽。本机（M5 Pro，5P+10E）实测：6 线程 8.09ms/tok、8 线程 6.99、
    //       10 线程 8.98（互踩）——8 = 5+3 最优。
    inline int default_parallelism() {
      // 优先检查环境变量覆盖（调优/实验用）
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
        p = perf_cores + 3; // P 核数 + 3 个额外线程（利用 E 核挤带宽）
      }
#endif
      // 非 Apple 平台或 sysctl 失败时，退回全部硬件线程数
      if (p <= 0) p = static_cast<int>(std::thread::hardware_concurrency());
      if (p <= 1) p = 2; // 至少 2 线程（单核时多线程无意义，靠内联路径兜底）
      return p > 16 ? 16 : p; // 上限 16
    }

    // ========================================================================
    // RowPool — 常驻行切分线程池
    // ========================================================================
    // 设计要点：
    //   - 常驻线程：decode 每个 token 要调 ~170 次 matvec，现起线程的创建开销
    //     会比小矩阵本身还贵。worker 启动后自旋待命，全程无 OS 锁、无系统调用。
    //   - fork-join：master 写任务参数 → release 递增 job_gen → 各 worker acquire
    //     到后代号算自己的行块 → done_gen.fetch_add 归位 → master 自己也算第 0 块，
    //     然后自旋等 done_gen 达标后返回。
    //   - 支持三种模式：kSingle（单矩阵）、kPair（两矩阵成对）、kQkv（Q/K/V 三矩阵）
    struct RowPool {
      // ---- 任务参数（仅 master 在发布新任务前写；worker 只在 acquire 后读）----
      const float16_t *w = nullptr;   // 第一个矩阵（或单矩阵模式的唯一矩阵）
      const float16_t *w2 = nullptr;  // 第二个矩阵（pair/qkv 模式）
      const float16_t *w3 = nullptr;  // 第三个矩阵（qkv 模式的 V 矩阵）
      const float *x = nullptr;       // 共享输入向量
      float *y = nullptr;             // 第一个输出
      float *y2 = nullptr;            // 第二个输出（pair/qkv 模式）
      float *y3 = nullptr;            // 第三个输出（qkv 模式）
      int out_dim = 0;                // 单矩阵的行数
      int in_dim = 0;                 // 输入维度（列数）
      int qkv_q_dim = 0;              // qkv 模式：Q 矩阵的行数
      int qkv_kv_dim = 0;             // qkv 模式：K/V 矩阵各自的行数
      enum Mode { kSingle, kPair, kQkv } mode = kSingle; // 当前任务模式
      bool pair = false;              // 是否是成对模式（兼容旧逻辑）

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
          // 每个 worker 绑定自己的 idx，进入 worker_main 循环
          workers.emplace_back([this, idx] { worker_main(idx); });
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
      // 参数：idx — 线程编号（0 是 master，1..p-1 是 worker）
      // 说明：将总行数均分给 p 个线程，前 rem 块各多分 1 行。
      //       根据 mode 决定从哪个矩阵读行、写到哪个输出。
      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1; // 总并行度
        int total; // 需要处理的总行数
        if (mode == kQkv) total = qkv_q_dim + 2 * qkv_kv_dim; // QKV：Q行 + K行 + V行
        else total = pair ? 2 * out_dim : out_dim; // pair：两矩阵行数之和
        // 均分：base = total/p，余数 rem = total%p
        const int base = total / p;
        const int rem = total % p;
        // 线程 idx 的行区间 [begin, end)
        // 前 rem 个线程各多分 1 行，后面的线程分 base 行
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        // 遍历分配给本线程的每一行
        for (int r = begin; r < end; ++r) {
          const float16_t *wm; // 当前行所属的权重矩阵
          float *ym;           // 当前行对应的输出向量
          int o;               // 行在所属矩阵内的偏移
          if (mode == kQkv) {
            // QKV 模式：按行号判断属于 Q/K/V 哪个矩阵
            if (r < qkv_q_dim) {
              o = r; wm = w; ym = y; // Q 矩阵
            } else if (r < qkv_q_dim + qkv_kv_dim) {
              o = r - qkv_q_dim; wm = w2; ym = y2; // K 矩阵
            } else {
              o = r - qkv_q_dim - qkv_kv_dim; wm = w3; ym = y3; // V 矩阵
            }
          } else if (pair && r >= out_dim) {
            // pair 模式：后半部分属于第二个矩阵
            o = r - out_dim; wm = w2; ym = y2;
          } else {
            // 单矩阵模式或 pair 的前半部分
            o = r; wm = w; ym = y;
          }
          // 定位行起点并计算点积
          const float16_t *row = wm + static_cast<size_t>(o) * in_dim;
          ym[o] = dot_row(row, x, in_dim);
        }
      }

      // fork-join 发布入口（仅 master 调用）
      // 流程：递增 job_gen 发布任务 → master 自己算第 0 块 → 等所有 worker 完成
      void publish_and_run() {
        job_gen.store(++job_counter, std::memory_order_release); // 发布新任务
        do_chunk(0); // master 自己算第 0 块
        expected_done += workers.size(); // 预期完成的 worker 数累加
        spin_until(done_gen, expected_done); // 等待所有 worker 完成
      }

      // 单矩阵模式入口
      void run(const float16_t *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_;       // 设置权重矩阵
        w2 = nullptr; // 清除第二矩阵指针
        x = x_;       // 设置输入向量
        y = y_;       // 设置输出向量
        y2 = nullptr;
        out_dim = out_dim_;
        in_dim = in_dim_;
        pair = false;
        mode = kSingle; // 标记为单矩阵模式
        publish_and_run();
      }

      // 成对模式入口（k/v 融合：两矩阵共享同一输入向量）
      void run_pair(const float16_t *w1_, const float16_t *w2_, const float *x_, float *y1_,
                    float *y2_, int out_dim_, int in_dim_) {
        w = w1_;      // 第一个矩阵
        w2 = w2_;     // 第二个矩阵
        x = x_;       // 共享输入
        y = y1_;      // 第一个输出
        y2 = y2_;     // 第二个输出
        out_dim = out_dim_;
        in_dim = in_dim_;
        pair = true;
        mode = kPair; // 标记为成对模式
        publish_and_run();
      }

      // QKV 三矩阵融合入口（Q/K/V 投影一步完成）
      void run_qkv(const float16_t *wq_, const float16_t *wk_, const float16_t *wv_,
                   const float *x_, float *yq_, float *yk_, float *yv_,
                   int q_dim_, int kv_dim_, int in_dim_) {
        w = wq_;       // Q 矩阵
        w2 = wk_;      // K 矩阵
        w3 = wv_;      // V 矩阵
        x = x_;        // 共享输入
        y = yq_;       // Q 输出
        y2 = yk_;      // K 输出
        y3 = yv_;      // V 输出
        qkv_q_dim = q_dim_;   // Q 的行数
        qkv_kv_dim = kv_dim_; // K/V 各自的行数
        in_dim = in_dim_;
        pair = false;
        mode = kQkv;   // 标记为 QKV 模式
        publish_and_run();
      }
    };

    // Meyers singleton：首次调用时创建线程池，进程退出时自动析构
    RowPool &pool() {
      static RowPool p; // C++11 magic static，线程安全
      return p;
    }

    // 粒度阈值（按元素数计）：权重不足该值时不值得并行（同步开销 > 收益）
    // 262144 float ≈ 1MB：q/o_proj（896×896，3.2MB）走并行；
    // k/v_proj（128×896，0.45MB）走内联
    constexpr std::size_t kMinParallelElems = 262144;
    // pair 粒度阈值：两矩阵元素总和 < 0.5MB 时不值得并行
    constexpr std::size_t kMinPairParallelElems = 131072;

    // 单线程内联 matvec：直接在当前线程逐行计算（不走线程池）
    inline void matvec_inline(const float16_t *w, const float *x, float *y, int out_dim,
                              int in_dim) {
      for (int o = 0; o < out_dim; ++o) {
        const float16_t *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row(row, x, in_dim);
      }
    }
  } // namespace

  // =========================================================================
  // matvec_f16_neon_mt_kv_nt() — 单矩阵 f16 matvec 外部入口
  // =========================================================================
  // 功能：计算 y = W @ x（f16 权重，NEON + 多线程 + LDNP）
  // 参数：
  //   w        — f16 权重矩阵（以 uint16_t* 传入，内部 reinterpret 为 float16_t*）
  //   x        — fp32 输入向量
  //   y        — fp32 输出向量
  //   out_dim  — 输出维度（W 的行数）
  //   in_dim   — 输入维度（W 的列数 = x 的长度）
  // 说明：权重元素总量低于阈值时走单线程内联路径，否则提交给线程池并行。
  void matvec_f16_neon_mt_kv_nt(const uint16_t *w, const float *x, float *y, int out_dim,
                                int in_dim) {
    RowPool &p = pool(); // 获取全局线程池
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim; // 权重总元素数
    // 小矩阵或无线程可用时走单线程路径
    if (elems < kMinParallelElems || p.workers.empty()) {
      matvec_inline(reinterpret_cast<const float16_t *>(w), x, y, out_dim, in_dim);
      return;
    }
    // 大矩阵走多线程路径
    p.run(reinterpret_cast<const float16_t *>(w), x, y, out_dim, in_dim);
  }

  // =========================================================================
  // matvec_pair_f16_neon_mt_kv_nt() — 成对 f16 matvec 外部入口（k/v 融合）
  // =========================================================================
  // 功能：同时计算 y1 = W1 @ x 和 y2 = W2 @ x（共享输入向量，一次 fork-join）
  // 参数：
  //   w1, w2  — 两个 f16 权重矩阵
  //   x       — 共享 fp32 输入向量
  //   y1, y2  — 两个 fp32 输出向量
  //   out_dim — 每个矩阵的行数
  //   in_dim  — 输入维度
  void matvec_pair_f16_neon_mt_kv_nt(const uint16_t *w1, const uint16_t *w2, const float *x,
                                     float *y1, float *y2, int out_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    // 两矩阵元素总和低于阈值时分别走单线程
    if (2 * elems < kMinPairParallelElems || p.workers.empty()) {
      matvec_inline(reinterpret_cast<const float16_t *>(w1), x, y1, out_dim, in_dim);
      matvec_inline(reinterpret_cast<const float16_t *>(w2), x, y2, out_dim, in_dim);
      return;
    }
    // 够大则合并成一次 fork-join
    p.run_pair(reinterpret_cast<const float16_t *>(w1), reinterpret_cast<const float16_t *>(w2),
               x, y1, y2, out_dim, in_dim);
  }

  // =========================================================================
  // matvec_qkv_f16_neon_mt_kv_nt() — QKV 三矩阵融合 f16 matvec 外部入口
  // =========================================================================
  // 功能：同时计算 yq = Wq @ x, yk = Wk @ x, yv = Wv @ x（一次 fork-join）
  // 参数：
  //   wq, wk, wv — Q/K/V 三个 f16 权重矩阵
  //   x          — 共享 fp32 输入向量
  //   yq, yk, yv — Q/K/V 三个 fp32 输出向量
  //   q_dim      — Q 矩阵的行数
  //   kv_dim     — K/V 矩阵各自的行数
  //   in_dim     — 输入维度
  void matvec_qkv_f16_neon_mt_kv_nt(const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
                                    const float *x, float *yq, float *yk, float *yv,
                                    int q_dim, int kv_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t total_rows = static_cast<size_t>(q_dim) + 2 * kv_dim; // Q行 + K行 + V行
    // 总行数 × in_dim 低于阈值时分别走单线程
    if (total_rows * in_dim < kMinParallelElems || p.workers.empty()) {
      matvec_inline(reinterpret_cast<const float16_t *>(wq), x, yq, q_dim, in_dim);
      matvec_inline(reinterpret_cast<const float16_t *>(wk), x, yk, kv_dim, in_dim);
      matvec_inline(reinterpret_cast<const float16_t *>(wv), x, yv, kv_dim, in_dim);
      return;
    }
    // 够大则三矩阵合并成一次 fork-join
    p.run_qkv(reinterpret_cast<const float16_t *>(wq),
              reinterpret_cast<const float16_t *>(wk),
              reinterpret_cast<const float16_t *>(wv),
              x, yq, yk, yv, q_dim, kv_dim, in_dim);
  }

  // 自注册进 f16 注册表：与 f32 阶梯顶层同名 "neon_mt_kv_nt"，
  // 按模型 dtype 解析（f16 模型选到本实现）。仅 aarch64 构建存在。
  TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_neon_mt_kv_nt, "neon_mt_kv_nt");
  TINYQWEN_MATVEC_F16_PAIR_VARIANT(matvec_pair_f16_neon_mt_kv_nt, "neon_mt_kv_nt");
  TINYQWEN_MATVEC_QKV_F16_VARIANT(matvec_qkv_f16_neon_mt_kv_nt, "neon_mt_kv_nt");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
