// bench_sustain.cpp — 持续负载热衰减曲线（测量项 6）
//
// 全部线程做流式读，持续 duration 秒，每 window 秒报告一次 GB/s，
// 观察热降频导致的吞吐衰减。
//
// 用法：./bench_sustain [threads=10] [duration_s=300] [window_s=10] [buffer_mb=2048]
// 编译：clang++ -O3 -std=c++17 bench_sustain.cpp -o bench_sustain
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

static uint64_t read_pass(const uint8_t* p, size_t n) {
  uint64_t acc = 0;
  size_t i = 0;
  for (; i + 128 <= n; i += 128) {
    uint64x2_t v0 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 0));
    uint64x2_t v1 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 16));
    uint64x2_t v2 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 32));
    uint64x2_t v3 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 48));
    uint64x2_t v4 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 64));
    uint64x2_t v5 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 80));
    uint64x2_t v6 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 96));
    uint64x2_t v7 = vld1q_u64(reinterpret_cast<const uint64_t*>(p + i + 112));
    acc += vgetq_lane_u64(v0, 0) + vgetq_lane_u64(v1, 1) +
           vgetq_lane_u64(v2, 0) + vgetq_lane_u64(v3, 1) +
           vgetq_lane_u64(v4, 0) + vgetq_lane_u64(v5, 1) +
           vgetq_lane_u64(v6, 0) + vgetq_lane_u64(v7, 1);
  }
  for (; i < n; ++i) acc += p[i];
  return acc;
}

struct Job {
  uint8_t* buf;
  size_t total;
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> bytes{0};
  std::atomic<uint64_t> sink{0};
};

static void worker(int t, int threads, Job* j) {
  size_t seg = j->total / threads;
  size_t off = (size_t)t * seg;
  size_t n = (t == threads - 1) ? (j->total - off) : seg;
  uint8_t* p = j->buf + off;
  uint64_t local = 0, b = 0;
  while (!j->start.load(std::memory_order_acquire)) {
  }
  while (!j->stop.load(std::memory_order_relaxed)) {
    local += read_pass(p, n);
    b += n;
    if (b >= (1ull << 30)) {  // 周期性刷回全局计数
      j->bytes.fetch_add(b);
      b = 0;
    }
  }
  j->bytes.fetch_add(b);
  j->sink.fetch_add(local);
}

int main(int argc, char** argv) {
  int threads = (argc > 1) ? atoi(argv[1]) : 10;
  double duration = (argc > 2) ? atof(argv[2]) : 300.0;
  double window = (argc > 3) ? atof(argv[3]) : 10.0;
  size_t buffer_mb = (argc > 4) ? atol(argv[4]) : 2048;

  size_t total = buffer_mb << 20;
  uint8_t* buf = static_cast<uint8_t*>(malloc(total));
  if (!buf) { fprintf(stderr, "alloc fail\n"); return 1; }
  memset(buf, 1, total);

  Job j;
  j.buf = buf;
  j.total = total;
  std::vector<std::thread> ts;
  for (int t = 0; t < threads; ++t)
    ts.emplace_back(worker, t, threads, &j);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));

  printf("threads\t%d\nduration_s\t%.0f\nwindow_s\t%.0f\n", threads, duration,
         window);
  printf("elapsed_s\tGB/s\n");
  fflush(stdout);
  double t0 = now_s();
  j.start.store(true, std::memory_order_release);
  uint64_t prev = 0;
  double prev_el = 0;
  double next_report = window;
  while (now_s() - t0 < duration) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    double el = now_s() - t0;
    if (el >= next_report) {
      uint64_t cur = j.bytes.load();
      double gbps = (cur - prev) / (el - prev_el) / 1e9;
      printf("%.0f\t%.2f\n", el, gbps);
      fflush(stdout);
      prev = cur;
      prev_el = el;
      next_report += window;
    }
  }
  j.stop.store(true);
  for (auto& t : ts) t.join();
  double total_el = now_s() - t0;
  printf("avg_total\t%.2f GB/s over %.0fs\n", j.bytes.load() / total_el / 1e9,
         total_el);
  if (j.sink.load() == 0xDEADBEEFull) fprintf(stderr, "?");
  free(buf);
  return 0;
}
