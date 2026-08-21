// ============================================================================
// matvec_f32_neon_mt_kv_nt.cpp — 矩阵乘向量：y = W @ x
//                  NEON + 多线程 + k/v 融合 + 流式加载（LDNP）版
// ============================================================================
// 在 neon_mt_kv 之上只改一件事：**权重加载指令**。
//
// ============================================================================
// 动机：decode 的权重是纯流式访问
// ============================================================================
// 每生成一个 token，约 2GB 权重被完整读一遍，每块权重只读一次、短期
// 内不再碰——教科书式的 streaming load。普通的 vld1q/ldp 加载会按常规
// 策略填 cache；LDNP（Load Non-temporal Pair）则给硬件一个提示：这对
// 数据"没有重访价值"，不必在 cache 里久留。理论收益：少污染 cache
// （x 向量和激活这些小而热的数据不被挤出去）、可能让带宽更顺。
//
// 预期收益≈0，仍然要做——这是 fp32 路线上最后一块可以排除的石头：
//   - Apple 核的硬件预取器极强，软件提示历史上常被无视；
//   - 归因阶梯里"撞带宽墙后算术类优化贡献≈0"（FMA 条目）的教训提示：
//     墙边的任何"计算侧/加载侧微调"都可能归零。测了才知道，测过才排除。
//
// ============================================================================
// 实现要点
// ============================================================================
// 1. clang 的 __builtin_nontemporal_load 对本工具的 aarch64 NEON 向量
//    加载**静默丢弃提示**（实测 codegen 仍出普通 ldp），所以 LDNP 用
//    内联汇编显式发射：`ldnp q0, q1, [ptr]` 一条指令加载 32B = 8 float，
//    顺带把加载指令数减半（16 float 从 4 条 ldp 变 2 条 ldnp）；
// 2. **LDNP 要求地址 16B 对齐**（SIMD&FP load pair 不对齐即 fault）。
//    真实模型里所有权重的行起点都满足对齐，但 kernel 不假设调用方，
//    逐行检查对齐：不对齐的行退回普通 vld1q 点积（与 neon 逐位一致）；
// 3. x 向量的加载**保持普通 vld1q**——x 每行都要重用（热数据），
//    恰恰要留在 cache 里，只有权重流适用非时间提示；
// 4. 除加载指令外，累加结构（4 链 FMA）、尾段处理、线程池、pair/QKV 融合
//    与 neon_mt_kv 完全一致，所以对齐门禁相同。
//
// 提供三个入口：单矩阵 / 成对(pair) / QKV 三矩阵融合。
// 选用：--matvec-impl neon_mt_kv_nt（仅 aarch64 构建注册；其他平台编译为空）。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT / _PAIR_VARIANT / _QKV_VARIANT
#include "ref_ops.h"

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#include <atomic>
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
    // ========================================================================
    // dot_row_neon() — 普通行点积：与 neon_mt 逐位一致（对齐兜底路径）
    // ========================================================================
    inline float dot_row_neon(const float *row, const float *x, int n) {
      float32x4_t acc0 = vdupq_n_f32(0.0f); // 累加器 0
      float32x4_t acc1 = vdupq_n_f32(0.0f); // 累加器 1
      float32x4_t acc2 = vdupq_n_f32(0.0f); // 累加器 2
      float32x4_t acc3 = vdupq_n_f32(0.0f); // 累加器 3
      int i = 0;
      const int n16 = n & ~15; // 主循环边界：16 的倍数
      for (; i < n16; i += 16) {
        // 4 条独立 FMA 链，各处理 4 个连续元素
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(row + i + 4), vld1q_f32(x + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(row + i + 8), vld1q_f32(x + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
      }
      const int n4 = n & ~3; // 向量尾段边界：4 的倍数
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(row + i), vld1q_f32(x + i));
      }
      // 合并 4 个累加器 + 横向归约
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      // 标量尾段
      for (; i < n; ++i) {
        total += row[i] * x[i];
      }
      return total;
    }

    // ========================================================================
    // ldnp_pair() — LDNP 流式加载一对 q 寄存器（32B = 8 float）
    // ========================================================================
    // 功能：用 LDNP 指令从地址 p 加载 32 字节到两个 128 位 NEON 寄存器，
    //       带非时间提示（不污染 cache）。
    // 参数：p — 源地址（必须 16B 对齐）, lo/hi — 输出寄存器
    // 说明：内联汇编中 %q 让编译器分配 q 寄存器编号；=&w 是 early-clobber
    //       约束，防止输出寄存器与输入指针寄存器冲突。
    inline void ldnp_pair(const float *p, float32x4_t &lo, float32x4_t &hi) {
      __asm__("ldnp %q[lo], %q[hi], [%[p]]"
              : [lo] "=&w"(lo), [hi] "=&w"(hi)
              : [p] "r"(p));
    }

    // ========================================================================
    // dot_row_neon_nt() — 流式行点积：权重走 LDNP，x 走普通加载
    // ========================================================================
    // 主循环每次迭代吃 32 个元素：4 条 LDNP（各 8 float）喂给 4 条累加链，
    // 每条链两次 FMA。尾段（<32）退回普通加载的向量/标量处理。
    inline float dot_row_neon_nt(const float *row, const float *x, int n) {
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      const int n32 = n & ~31; // 主循环边界：32 的倍数
      for (; i < n32; i += 32) {
        // 用 LDNP 加载 8 组各 4 个 float（共 32 float = 128 字节权重）
        float32x4_t w0, w1, w2, w3, w4, w5, w6, w7;
        ldnp_pair(row + i, w0, w1);      // LDNP: float[i..i+7]
        ldnp_pair(row + i + 8, w2, w3);  // LDNP: float[i+8..i+15]
        ldnp_pair(row + i + 16, w4, w5); // LDNP: float[i+16..i+23]
        ldnp_pair(row + i + 24, w6, w7); // LDNP: float[i+24..i+31]
        // 每条链消费 2 个 LDNP 结果，x 用普通 vld1q_f32
        acc0 = vfmaq_f32(acc0, w0, vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, w1, vld1q_f32(x + i + 4));
        acc1 = vfmaq_f32(acc1, w2, vld1q_f32(x + i + 8));
        acc1 = vfmaq_f32(acc1, w3, vld1q_f32(x + i + 12));
        acc2 = vfmaq_f32(acc2, w4, vld1q_f32(x + i + 16));
        acc2 = vfmaq_f32(acc2, w5, vld1q_f32(x + i + 20));
        acc3 = vfmaq_f32(acc3, w6, vld1q_f32(x + i + 24));
        acc3 = vfmaq_f32(acc3, w7, vld1q_f32(x + i + 28));
      }
      // 尾段：普通加载
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

    // 行点积入口：行起点 16B 对齐走 LDNP 路径，否则兜底普通路径
    inline float dot_row(const float *row, const float *x, int n) {
      if ((reinterpret_cast<std::uintptr_t>(row) & 15) != 0) {
        return dot_row_neon(row, x, n); // 未对齐：退回普通路径
      }
      return dot_row_neon_nt(row, x, n); // 对齐：走 LDNP
    }

    // 自旋等待
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

    // 默认并行度
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

    // ========================================================================
    // RowPool — 常驻行切分线程池（支持 single/pair/QKV 三种模式）
    // ========================================================================
    struct RowPool {
      const float *w = nullptr;
      const float *w2 = nullptr;
      const float *w3 = nullptr;
      const float *x = nullptr;
      float *y = nullptr;
      float *y2 = nullptr;
      float *y3 = nullptr;
      int out_dim = 0;
      int in_dim = 0;
      int qkv_q_dim = 0;
      int qkv_kv_dim = 0;
      enum Mode { kSingle, kPair, kQkv } mode = kSingle;
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
        for (auto &t : workers) { t.join(); }
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

      // 第 idx 块行区间上的逐行点积（根据 mode 选择矩阵和输出）
      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        int total;
        if (mode == kQkv) total = qkv_q_dim + 2 * qkv_kv_dim;
        else total = pair ? 2 * out_dim : out_dim;
        const int base = total / p;
        const int rem = total % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        for (int r = begin; r < end; ++r) {
          const float *wm; float *ym; int o;
          if (mode == kQkv) {
            if (r < qkv_q_dim) { o = r; wm = w; ym = y; }
            else if (r < qkv_q_dim + qkv_kv_dim) { o = r - qkv_q_dim; wm = w2; ym = y2; }
            else { o = r - qkv_q_dim - qkv_kv_dim; wm = w3; ym = y3; }
          } else if (pair && r >= out_dim) {
            o = r - out_dim; wm = w2; ym = y2;
          } else {
            o = r; wm = w; ym = y;
          }
          const float *row = wm + static_cast<size_t>(o) * in_dim;
          ym[o] = dot_row(row, x, in_dim);
        }
      }

      void publish_and_run() {
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
      }

      void run(const float *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
        w = w_; w2 = nullptr; x = x_; y = y_; y2 = nullptr;
        out_dim = out_dim_; in_dim = in_dim_; pair = false; mode = kSingle;
        publish_and_run();
      }

      void run_pair(const float *w1_, const float *w2_, const float *x_, float *y1_,
                    float *y2_, int out_dim_, int in_dim_) {
        w = w1_; w2 = w2_; x = x_; y = y1_; y2 = y2_;
        out_dim = out_dim_; in_dim = in_dim_; pair = true; mode = kPair;
        publish_and_run();
      }

      void run_qkv(const float *wq_, const float *wk_, const float *wv_,
                   const float *x_, float *yq_, float *yk_, float *yv_,
                   int q_dim_, int kv_dim_, int in_dim_) {
        w = wq_; w2 = wk_; w3 = wv_; x = x_; y = yq_; y2 = yk_; y3 = yv_;
        qkv_q_dim = q_dim_; qkv_kv_dim = kv_dim_; in_dim = in_dim_;
        pair = false; mode = kQkv;
        publish_and_run();
      }
    };

    RowPool &pool() { static RowPool p; return p; }

    constexpr std::size_t kMinParallelElems = 262144;
    constexpr std::size_t kMinPairParallelElems = 131072;

    inline void matvec_inline(const float *w, const float *x, float *y, int out_dim,
                              int in_dim) {
      for (int o = 0; o < out_dim; ++o) {
        const float *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row(row, x, in_dim);
      }
    }
  } // namespace

  // 单矩阵入口
  void matvec_f32_neon_mt_kv_nt(const float *w, const float *x, float *y, int out_dim,
                                int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (elems < kMinParallelElems || p.workers.empty()) {
      matvec_inline(w, x, y, out_dim, in_dim);
      return;
    }
    p.run(w, x, y, out_dim, in_dim);
  }

  // 成对入口
  void matvec_pair_f32_neon_mt_kv_nt(const float *w1, const float *w2, const float *x,
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

  // QKV 三矩阵融合入口
  void matvec_qkv_f32_neon_mt_kv_nt(const float *wq, const float *wk, const float *wv,
                                    const float *x, float *yq, float *yk, float *yv,
                                    int q_dim, int kv_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t total_rows = static_cast<size_t>(q_dim) + 2 * kv_dim;
    if (total_rows * in_dim < kMinParallelElems || p.workers.empty()) {
      matvec_inline(wq, x, yq, q_dim, in_dim);
      matvec_inline(wk, x, yk, kv_dim, in_dim);
      matvec_inline(wv, x, yv, kv_dim, in_dim);
      return;
    }
    p.run_qkv(wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim);
  }

  // 自注册：仅 aarch64 构建存在
  TINYQWEN_MATVEC_VARIANT(matvec_f32_neon_mt_kv_nt, "neon_mt_kv_nt");
  TINYQWEN_MATVEC_PAIR_VARIANT(matvec_pair_f32_neon_mt_kv_nt, "neon_mt_kv_nt");
  TINYQWEN_MATVEC_QKV_VARIANT(matvec_qkv_f32_neon_mt_kv_nt, "neon_mt_kv_nt");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
