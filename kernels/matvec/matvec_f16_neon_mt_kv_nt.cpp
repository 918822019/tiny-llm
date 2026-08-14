// 矩阵乘向量（f16 权重）：y = W @ x —— NEON + 多线程 + k/v 融合 + LDNP 版。
//
// f16 族的"满栈变体"：把 f32 阶梯顶层 neon_mt_kv_nt 的全部优化移植到
// 半精度权重上。权重是 IEEE binary16（每 token 的权重流量直接减半），
// x/y 保持 fp32（weight-only：只省搬运，计算精度不降）。
//
// 与 f32 版的逐项对应（行内数学语义不变，只是权重加载变窄）：
//   - 4 链 FMA 累加：每链用 vld1q_f16 一次载 8 个 half，vcvt 成两半
//     float32x4 后 FMA——累加仍在 fp32，不引入 fp16 累加的精度风险；
//   - 常驻线程池行切分、粒度阈值、k/v 成对融合：与 f32 版相同；
//   - LDNP 流式加载：权重按 ldnp q 对（32B = 16 half）非时间加载，
//     行 16B 对齐检查 + 非对齐兜底。真实模型里行步长 = in_dim×2，
//     896/4864/128 都是 8 的倍数 → 逐行 16B 对齐成立。
//
// f16 特有的数值约定：与 matvec_f16_ref（double 累加）的差只来自
// float 累加顺序，量级与 f32 族相同，门禁容差沿用 5e-3。
//
// 注册进 f16 注册表，实现名与 f32 阶梯顶层**同名** "neon_mt_kv_nt"——
// 选谁由模型文件的 dtype 决定（main 按 dtype 查表，见 dispatch.h 注释）。
// 仅 aarch64 构建注册；其他平台 f16 模型自动用 matvec_f16_ref 兜底。

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT / _F16_PAIR_VARIANT
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
    // ---- 普通行点积（f16 权重）：对齐兜底路径，与 nt 版数值一致 ----
    inline float dot_row_f16(const float16_t *row, const float *x, int n) {
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      // 主循环：一次 32 个 half（4 链 × 每链 vld1q_f16 载 8 个）。
      const int n32 = n & ~31;
      for (; i < n32; i += 32) {
        float16x8_t h0 = vld1q_f16(row + i);
        float16x8_t h1 = vld1q_f16(row + i + 8);
        float16x8_t h2 = vld1q_f16(row + i + 16);
        float16x8_t h3 = vld1q_f16(row + i + 24);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h0), vld1q_f32(x + i + 4));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + i + 8));
        acc1 = vfmaq_f32(acc1, vcvt_high_f32_f16(h1), vld1q_f32(x + i + 12));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(h2)), vld1q_f32(x + i + 16));
        acc2 = vfmaq_f32(acc2, vcvt_high_f32_f16(h2), vld1q_f32(x + i + 20));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_low_f16(h3)), vld1q_f32(x + i + 24));
        acc3 = vfmaq_f32(acc3, vcvt_high_f32_f16(h3), vld1q_f32(x + i + 28));
      }
      // 尾段：8 个一批 -> 4 个一批 -> 标量。
      const int n8 = n & ~7;
      for (; i < n8; i += 8) {
        float16x8_t h = vld1q_f16(row + i);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
      }
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16(row + i)), vld1q_f32(x + i));
      }
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      for (; i < n; ++i) {
        total += static_cast<float>(row[i]) * x[i];
      }
      return total;
    }

    // LDNP 一次读一对 q 寄存器（32B = 16 half），带非时间提示。
    inline void ldnp_pair_f16(const float16_t *p, float16x8_t &lo, float16x8_t &hi) {
      __asm__("ldnp %q[lo], %q[hi], [%[p]]"
              : [lo] "=&w"(lo), [hi] "=&w"(hi)
              : [p] "r"(p));
    }

    // ---- 流式行点积：权重走 LDNP，x 保持普通加载（热数据要留在 cache）----
    // 主循环一次 32 个 half = 2 条 LDNP（各载 16 half = 2 个 q）。
    inline float dot_row_f16_nt(const float16_t *row, const float *x, int n) {
      float32x4_t acc0 = vdupq_n_f32(0.0f);
      float32x4_t acc1 = vdupq_n_f32(0.0f);
      float32x4_t acc2 = vdupq_n_f32(0.0f);
      float32x4_t acc3 = vdupq_n_f32(0.0f);
      int i = 0;
      const int n32 = n & ~31;
      for (; i < n32; i += 32) {
        float16x8_t h0, h1, h2, h3;
        ldnp_pair_f16(row + i, h0, h1);      // half 0-15
        ldnp_pair_f16(row + i + 16, h2, h3); // half 16-31
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h0)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h0), vld1q_f32(x + i + 4));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_low_f16(h1)), vld1q_f32(x + i + 8));
        acc1 = vfmaq_f32(acc1, vcvt_high_f32_f16(h1), vld1q_f32(x + i + 12));
        acc2 = vfmaq_f32(acc2, vcvt_f32_f16(vget_low_f16(h2)), vld1q_f32(x + i + 16));
        acc2 = vfmaq_f32(acc2, vcvt_high_f32_f16(h2), vld1q_f32(x + i + 20));
        acc3 = vfmaq_f32(acc3, vcvt_f32_f16(vget_low_f16(h3)), vld1q_f32(x + i + 24));
        acc3 = vfmaq_f32(acc3, vcvt_high_f32_f16(h3), vld1q_f32(x + i + 28));
      }
      const int n8 = n & ~7;
      for (; i < n8; i += 8) {
        float16x8_t h = vld1q_f16(row + i);
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        acc0 = vfmaq_f32(acc0, vcvt_high_f32_f16(h), vld1q_f32(x + i + 4));
      }
      const int n4 = n & ~3;
      for (; i < n4; i += 4) {
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vld1_f16(row + i)), vld1q_f32(x + i));
      }
      const float32x4_t sum01 = vaddq_f32(acc0, acc1);
      const float32x4_t sum23 = vaddq_f32(acc2, acc3);
      float total = vaddvq_f32(vaddq_f32(sum01, sum23));
      for (; i < n; ++i) {
        total += static_cast<float>(row[i]) * x[i];
      }
      return total;
    }

    // 行点积入口：16B 对齐走 LDNP，否则兜底普通加载（LDNP 非对齐即 fault）。
    inline float dot_row(const float16_t *row, const float *x, int n) {
      if ((reinterpret_cast<std::uintptr_t>(row) & 15) != 0) {
        return dot_row_f16(row, x, n);
      }
      return dot_row_f16_nt(row, x, n);
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

    // 并行度默认 = P 核数 + 3（比 f32 版的 +1 更多）。原因：f16 把每 token
    // 权重流量减半，单核更早撞到的是"总带宽还没吃满"而不是"单核份额到顶"，
    // 多几个 E 核能再挤带宽。本机（M5 Pro，5P+10E）实测扫描：
    //   6 线程 8.09ms/tok、8 线程 6.99、10 线程 8.98（互踩）——8 = 5+3 最优。
    // 换机器可能不同，TINYQWEN_MT_THREADS 可覆盖。
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
        p = perf_cores + 3;
      }
#endif
      if (p <= 0) p = static_cast<int>(std::thread::hardware_concurrency());
      if (p <= 1) p = 2;
      return p > 16 ? 16 : p;
    }

    // 常驻行切分线程池：与 f32 版 neon_mt_kv_nt 完全相同（单矩阵 + pair）。
    struct RowPool {
      const float16_t *w = nullptr;
      const float16_t *w2 = nullptr;
      const float *x = nullptr;
      float *y = nullptr;
      float *y2 = nullptr;
      int out_dim = 0;
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

      void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int total = pair ? 2 * out_dim : out_dim;
        const int base = total / p;
        const int rem = total % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        for (int r = begin; r < end; ++r) {
          const int o = pair && r >= out_dim ? r - out_dim : r;
          const float16_t *wm = pair && r >= out_dim ? w2 : w;
          float *ym = pair && r >= out_dim ? y2 : y;
          const float16_t *row = wm + static_cast<size_t>(o) * in_dim;
          ym[o] = dot_row(row, x, in_dim);
        }
      }

      void publish_and_run() {
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
      }

      void run(const float16_t *w_, const float *x_, float *y_, int out_dim_, int in_dim_) {
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

      void run_pair(const float16_t *w1_, const float16_t *w2_, const float *x_, float *y1_,
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

    // 粒度阈值沿用 f32 版（按元素数计）：f16 字节减半后并行段更轻，
    // 但同步开销不变——阈值是否要调留给测量（TINYQWEN_MT_THREADS 同理）。
    constexpr std::size_t kMinParallelElems = 262144;
    constexpr std::size_t kMinPairParallelElems = 131072;

    inline void matvec_inline(const float16_t *w, const float *x, float *y, int out_dim,
                              int in_dim) {
      for (int o = 0; o < out_dim; ++o) {
        const float16_t *row = w + static_cast<size_t>(o) * in_dim;
        y[o] = dot_row(row, x, in_dim);
      }
    }
  } // namespace

  void matvec_f16_neon_mt_kv_nt(const uint16_t *w, const float *x, float *y, int out_dim,
                                int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (elems < kMinParallelElems || p.workers.empty()) {
      matvec_inline(reinterpret_cast<const float16_t *>(w), x, y, out_dim, in_dim);
      return;
    }
    p.run(reinterpret_cast<const float16_t *>(w), x, y, out_dim, in_dim);
  }

  void matvec_pair_f16_neon_mt_kv_nt(const uint16_t *w1, const uint16_t *w2, const float *x,
                                     float *y1, float *y2, int out_dim, int in_dim) {
    RowPool &p = pool();
    const std::size_t elems = static_cast<size_t>(out_dim) * in_dim;
    if (2 * elems < kMinPairParallelElems || p.workers.empty()) {
      matvec_inline(reinterpret_cast<const float16_t *>(w1), x, y1, out_dim, in_dim);
      matvec_inline(reinterpret_cast<const float16_t *>(w2), x, y2, out_dim, in_dim);
      return;
    }
    p.run_pair(reinterpret_cast<const float16_t *>(w1), reinterpret_cast<const float16_t *>(w2),
               x, y1, y2, out_dim, in_dim);
  }

  // 自注册进 f16 注册表：与 f32 阶梯顶层同名 "neon_mt_kv_nt"，
  // 按模型 dtype 解析（f16 模型选到本实现）。仅 aarch64 构建存在。
  TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_neon_mt_kv_nt, "neon_mt_kv_nt");
  TINYQWEN_MATVEC_F16_PAIR_VARIANT(matvec_pair_f16_neon_mt_kv_nt, "neon_mt_kv_nt");
} // namespace tinyqwen

#endif // defined(__aarch64__) || defined(_M_ARM64)
