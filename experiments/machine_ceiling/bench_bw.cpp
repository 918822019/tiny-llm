// bench_bw.cpp — CPU 流式内存带宽 benchmark（机器极限账表 · 测量项 2）
//
// 目的：测 Apple Silicon 上 CPU 侧可达的 DRAM 流式带宽饱和曲线：
//   - read：纯流式读（decode 权重读取的模型）
//   - copy：读+写（带宽总线双向压力）
//   - 多线程 1..N 扫描，找到"几个核吃满总线"的拐点
//
// 用法：
//   ./bench_bw                          # 默认 2048MB buffer，全核扫描，每点 2s
//   ./bench_bw <buffer_mb> <max_threads> <duration_s>
//
// 编译：clang++ -O3 -std=c++17 bench_bw.cpp -o bench_bw
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

// 流式读一段内存，返回校验和（防止编译器消除 load）。
// 每次迭代读 128B（Apple Silicon cache line 大小），8 条 NEON 16B load。
// 用 8 条独立累加链，避免 loop-carried 依赖压低 MLP。
static uint64_t read_pass(const uint8_t* p, size_t n) {
  uint64x2_t a0 = vdupq_n_u64(0), a1 = vdupq_n_u64(0),
             a2 = vdupq_n_u64(0), a3 = vdupq_n_u64(0);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    a0 = vaddq_u64(
        a0, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 0)));
    a1 = vaddq_u64(
        a1, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 16)));
    a2 = vaddq_u64(
        a2, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 32)));
    a3 = vaddq_u64(
        a3, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 48)));
    a0 = vaddq_u64(
        a0, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 64)));
    a1 = vaddq_u64(
        a1, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 80)));
    a2 = vaddq_u64(
        a2, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 96)));
    a3 = vaddq_u64(
        a3, vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 112)));
  }
  uint64_t acc = vgetq_lane_u64(a0, 0) + vgetq_lane_u64(a0, 1) +
                 vgetq_lane_u64(a1, 0) + vgetq_lane_u64(a1, 1) +
                 vgetq_lane_u64(a2, 0) + vgetq_lane_u64(a2, 1) +
                 vgetq_lane_u64(a3, 0) + vgetq_lane_u64(a3, 1);
  for (; i < n; ++i) acc += p[i];
  return acc;
}

// 流式写一段内存。
static uint64_t write_pass(uint8_t* p, size_t n, uint8_t v) {
  uint8x16_t vv = vdupq_n_u8(v);
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    vst1q_u8(p + i + 0, vv);
    vst1q_u8(p + i + 16, vv);
    vst1q_u8(p + i + 32, vv);
    vst1q_u8(p + i + 48, vv);
    vst1q_u8(p + i + 64, vv);
    vst1q_u8(p + i + 80, vv);
    vst1q_u8(p + i + 96, vv);
    vst1q_u8(p + i + 112, vv);
  }
  for (; i < n; ++i) p[i] = v;
  return n;
}

// 拷贝：src -> dst（读+写各一遍）。
static uint64_t copy_pass(const uint8_t* src, uint8_t* dst, size_t n) {
  size_t i = 0;
  uint64_t acc = 0;
  for (; i + 128 <= n; i += 128) {
    uint8x16_t v0 = vld1q_u8(src + i + 0);
    uint8x16_t v1 = vld1q_u8(src + i + 16);
    uint8x16_t v2 = vld1q_u8(src + i + 32);
    uint8x16_t v3 = vld1q_u8(src + i + 48);
    uint8x16_t v4 = vld1q_u8(src + i + 64);
    uint8x16_t v5 = vld1q_u8(src + i + 80);
    uint8x16_t v6 = vld1q_u8(src + i + 96);
    uint8x16_t v7 = vld1q_u8(src + i + 112);
    vst1q_u8(dst + i + 0, v0);
    vst1q_u8(dst + i + 16, v1);
    vst1q_u8(dst + i + 32, v2);
    vst1q_u8(dst + i + 48, v3);
    vst1q_u8(dst + i + 64, v4);
    vst1q_u8(dst + i + 80, v5);
    vst1q_u8(dst + i + 96, v6);
    vst1q_u8(dst + i + 112, v7);
  }
  acc += i;
  return acc;
}

struct BenchArgs {
  uint8_t* buf;
  const uint8_t* src;  // copy 模式用（buf 前半 + 后半）
  size_t total;
  double stop_time;
  std::atomic<uint64_t> sink{0};
  std::atomic<uint64_t> bytes_done{0};
  std::atomic<bool> start{false};
};

static void worker(int t, int threads, const char* mode, BenchArgs* a) {
  size_t seg = a->total / threads;
  size_t off = static_cast<size_t>(t) * seg;
  size_t n = (t == threads - 1) ? (a->total - off) : seg;
  uint8_t* p = a->buf + off;
  uint64_t local = 0;
  uint64_t bytes = 0;
  while (!a->start.load(std::memory_order_acquire)) {
  }
  double stop = a->stop_time;
  if (strcmp(mode, "read") == 0) {
    do {
      local += read_pass(p, n);
      bytes += n;
    } while (now_s() < stop);
  } else if (strcmp(mode, "write") == 0) {
    do {
      local += write_pass(p, n, 0x5A);
      bytes += n;
    } while (now_s() < stop);
  } else {  // copy：段内前半读、后半写
    size_t half = n / 2;
    const uint8_t* s = p;
    uint8_t* d = p + half;
    do {
      local += copy_pass(s, d, half);
      bytes += half * 2;  // 读+写各计一半流量
    } while (now_s() < stop);
  }
  a->sink.fetch_add(local);
  a->bytes_done.fetch_add(bytes);
}

static double bench(const char* mode, uint8_t* buf, size_t total, int threads,
                    double duration) {
  BenchArgs a;
  a.buf = buf;
  a.total = total;
  double wall0 = now_s();
  a.stop_time = wall0 + duration;
  std::vector<std::thread> ts;
  ts.reserve(threads);
  for (int t = 0; t < threads; ++t)
    ts.emplace_back(worker, t, threads, mode, &a);
  // 让所有线程就位再放行
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  double start_t = now_s();
  a.stop_time = start_t + duration;
  a.start.store(true, std::memory_order_release);
  for (auto& t : ts) t.join();
  double wall1 = now_s();
  double gbps = static_cast<double>(a.bytes_done.load()) / (wall1 - start_t) /
                1e9;
  // 防优化
  if (a.sink.load() == 0xDEADBEEFull) fprintf(stderr, "?");
  return gbps;
}

int main(int argc, char** argv) {
  size_t buffer_mb = 2048;
  int max_threads = static_cast<int>(std::thread::hardware_concurrency());
  double duration = 2.0;
  const char* mode = "read";
  if (argc > 1) buffer_mb = atol(argv[1]);
  if (argc > 2) max_threads = atoi(argv[2]);
  if (argc > 3) duration = atof(argv[3]);
  if (argc > 4) mode = argv[4];

  size_t total = buffer_mb * 1024ull * 1024ull;
  fprintf(stderr,
          "# bench_bw: mode=%s buffer=%zuMB threads=1..%d duration=%.1fs\n",
          mode, buffer_mb, max_threads, duration);
  uint8_t* buf = static_cast<uint8_t*>(malloc(total));
  if (!buf) {
    fprintf(stderr, "alloc failed\n");
    return 1;
  }
  memset(buf, 1, total);  // fault in 全部页面

  printf("threads\tGB/s\n");
  for (int t = 1; t <= max_threads; ++t) {
    double g = bench(mode, buf, total, t, duration);
    printf("%d\t%.2f\n", t, g);
    fflush(stdout);
  }
  free(buf);
  return 0;
}
