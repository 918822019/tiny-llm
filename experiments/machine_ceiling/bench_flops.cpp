// bench_flops.cpp — CPU 计算上限（测量项 4）
//
// 模式：
//   f32   ：NEON fp32 FMA 峰值（GFLOPS）——decode 里 matvec 算力头的天花板
//   i8    ：NEON dotprod (vdotq_s32) 峰值（GOPS）——INT4/W4A8/SDOT 路线相关
//   gemm  ：Accelerate cblas_sgemm（打 AMX 协处理器）——prefill GEMM 天花板
//
// 用法：./bench_flops <mode> [threads] [duration_s] [gemm_n]
// 编译：clang++ -O3 -std=c++17 -march=armv8.2-a+dotprod bench_flops.cpp \
//          -framework Accelerate -o bench_flops
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// ---- fp32 FMA：8 条独立累加链，打满 NEON 发射 ----
static double f32_gflops_core(double duration) {
  float32x4_t a0 = vdupq_n_f32(1.0f), a1 = vdupq_n_f32(1.1f),
              a2 = vdupq_n_f32(1.2f), a3 = vdupq_n_f32(1.3f),
              a4 = vdupq_n_f32(1.4f), a5 = vdupq_n_f32(1.5f),
              a6 = vdupq_n_f32(1.6f), a7 = vdupq_n_f32(1.7f);
  float32x4_t b = vdupq_n_f32(0.999999f), c = vdupq_n_f32(0.5f);
  double t0 = now_s();
  uint64_t iters = 0;
  do {
    for (int k = 0; k < 1024; ++k) {
      a0 = vfmaq_f32(a0, b, c);
      a1 = vfmaq_f32(a1, b, c);
      a2 = vfmaq_f32(a2, b, c);
      a3 = vfmaq_f32(a3, b, c);
      a4 = vfmaq_f32(a4, b, c);
      a5 = vfmaq_f32(a5, b, c);
      a6 = vfmaq_f32(a6, b, c);
      a7 = vfmaq_f32(a7, b, c);
    }
    iters += 1024;
  } while (now_s() - t0 < duration);
  double dt = now_s() - t0;
  double flops = (double)iters * 8 /*acc*/ * 8 /*flops per vfma f32x4*/;
  // 防止被优化掉
  volatile float sink = a0[0] + a1[1] + a2[2] + a3[3] + a4[0] + a5[1] +
                        a6[2] + a7[3];
  (void)sink;
  return flops / dt / 1e9;
}

// ---- i8 dotprod：独立累加链 ----
static double i8_gops_core(double duration) {
  int8x16_t x = vdupq_n_s8(1), y = vdupq_n_s8(2);
  int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0),
            a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
  double t0 = now_s();
  uint64_t iters = 0;
  do {
    for (int k = 0; k < 1024; ++k) {
      a0 = vdotq_s32(a0, x, y);
      a1 = vdotq_s32(a1, x, y);
      a2 = vdotq_s32(a2, x, y);
      a3 = vdotq_s32(a3, x, y);
    }
    iters += 1024;
  } while (now_s() - t0 < duration);
  double dt = now_s() - t0;
  // 每条 vdotq_s32：16 次乘 + 16 次加 = 32 int ops
  double ops = (double)iters * 4 /*acc*/ * 32;
  volatile int sink = a0[0] + a1[1] + a2[2] + a3[3];
  (void)sink;
  return ops / dt / 1e9;
}

struct Job {
  double duration;
  double (*fn)(double);
  std::atomic<bool> start{false};
  double stop_time;
  std::atomic<uint64_t> done_flag{0};
  std::vector<double> results;
};

static void worker(int t, Job* j) {
  while (!j->start.load(std::memory_order_acquire)) {
  }
  double g = j->fn(j->duration);
  j->results[t] = g;
  j->done_flag.fetch_add(1);
}

static double bench_mt(const char* mode, int threads, double duration) {
  Job j;
  j.duration = duration;
  j.fn = (strcmp(mode, "f32") == 0) ? f32_gflops_core : i8_gops_core;
  j.results.assign(threads, 0.0);
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t) ts.emplace_back(worker, t, &j);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  j.start.store(true, std::memory_order_release);
  for (auto& t : ts) t.join();
  double sum = 0;
  for (double g : j.results) sum += g;
  return sum;
}

// ---- AMX via Accelerate sgemm ----
static double gemm_gflops(int n, int repeats) {
  std::vector<float> A((size_t)n * n, 0.5f), B((size_t)n * n, 0.25f),
      C((size_t)n * n, 0.0f);
  // 预热
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
              A.data(), n, B.data(), n, 0.0f, C.data(), n);
  double t0 = now_s();
  for (int r = 0; r < repeats; ++r) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f,
                A.data(), n, B.data(), n, 0.0f, C.data(), n);
  }
  double dt = now_s() - t0;
  double flops = 2.0 * (double)n * n * n * repeats;
  volatile float sink = C[0] + C[n * n - 1];
  (void)sink;
  return flops / dt / 1e9;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s f32|i8|gemm [threads] [duration_s] [gemm_n]\n",
            argv[0]);
    return 1;
  }
  const char* mode = argv[1];
  if (strcmp(mode, "gemm") == 0) {
    int n = (argc > 2) ? atoi(argv[2]) : 2048;
    int repeats = (argc > 3) ? atoi(argv[3]) : 5;
    double g = gemm_gflops(n, repeats);
    printf("gemm_n\t%d\nGFLOPS\t%.1f\n", n, g);
    return 0;
  }
  int threads = (argc > 2) ? atoi(argv[2]) : 1;
  double duration = (argc > 3) ? atof(argv[3]) : 2.0;
  double g = bench_mt(mode, threads, duration);
  printf("mode\t%s\nthreads\t%d\n%s\t%.1f\n", mode, threads,
         strcmp(mode, "f32") == 0 ? "GFLOPS" : "GOPS", g);
  return 0;
}
