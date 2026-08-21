// bench_gpu_bw.mm — Metal GPU 流式带宽 + CPU/GPU 并发总带宽（测量项 3）
//
// 用法：
//   ./bench_gpu_bw gpu [buffer_mb] [duration_s]
//   ./bench_gpu_bw concurrent <cpu_threads> [buffer_mb] [duration_s]
//
// 编译：clang++ -O3 -std=c++17 bench_gpu_bw.mm -framework Metal -framework Foundation -o bench_gpu_bw
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

// 本机 Metal 运行时（Metal 4）暴露的是 computeCommandEncoder 而非 computeEncoder。
@interface NSObject (MachineCeilingCompat)
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
@end

static const char* kShaderSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void stream_read(device const float4* buf [[buffer(0)]],\n"
    "                        device float4* out [[buffer(1)]],\n"
    "                        constant uint& n_elems [[buffer(2)]],\n"
    "                        uint tid [[thread_position_in_grid]],\n"
    "                        uint gsize [[threads_per_grid]]) {\n"
    "  float4 acc = float4(0.0f);\n"
    "  for (uint i = tid; i < n_elems; i += gsize) { acc += buf[i]; }\n"
    "  if (acc.x == 12345.678f) out[tid] = acc;\n"
    "}\n";

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

struct CpuJob {
  uint8_t* buf;
  size_t total;
  double stop_time;
  std::atomic<bool> start{false};
  std::atomic<uint64_t> bytes_done{0};
  std::atomic<uint64_t> sink{0};
};

static void cpu_worker(int t, int threads, CpuJob* j) {
  size_t seg = j->total / threads;
  size_t off = static_cast<size_t>(t) * seg;
  size_t n = (t == threads - 1) ? (j->total - off) : seg;
  uint8_t* p = j->buf + off;
  uint64_t local = 0, bytes = 0;
  while (!j->start.load(std::memory_order_acquire)) {
  }
  double stop = j->stop_time;
  do {
    local += read_pass(p, n);
    bytes += n;
  } while (now_s() < stop);
  j->sink.fetch_add(local);
  j->bytes_done.fetch_add(bytes);
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s gpu|concurrent <cpu_threads?> [buffer_mb] [duration_s]\n", argv[0]);
    return 1;
  }
  bool concurrent = strcmp(argv[1], "concurrent") == 0;
  int cpu_threads = 0;
  int argi = 2;
  if (concurrent) {
    if (argc < 3) { fprintf(stderr, "need cpu_threads\n"); return 1; }
    cpu_threads = atoi(argv[2]);
    argi = 3;
  }
  size_t buffer_mb = (argc > argi) ? atol(argv[argi]) : 1024;
  double duration = (argc > argi + 1) ? atof(argv[argi + 1]) : 4.0;

  NSError* err = nil;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 1; }
  fprintf(stderr, "# device: %s, maxBufferLength=%lluMB\n",
          dev.name.UTF8String,
          (unsigned long long)(dev.maxBufferLength >> 20));
  if (buffer_mb > (dev.maxBufferLength >> 20))
    buffer_mb = dev.maxBufferLength >> 20;

  id<MTLLibrary> lib =
      [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                        options:nil
                          error:&err];
  if (!lib) {
    fprintf(stderr, "shader compile failed: %s\n",
            err ? err.localizedDescription.UTF8String : "(no error info)");
    return 1;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:@"stream_read"];
  id<MTLComputePipelineState> pso =
      [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) { fprintf(stderr, "pso failed\n"); return 1; }

  size_t bytes = buffer_mb << 20;
  uint32_t n_elems = (uint32_t)(bytes / 16);
  id<MTLBuffer> buf =
      [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  memset(buf.contents, 1, bytes);

  int gsize = 262144;
  id<MTLBuffer> out =
      [dev newBufferWithLength:(size_t)gsize * 16
                       options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q = [dev newCommandQueue];

  for (int i = 0; i < 2; ++i) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n_elems length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(gsize, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
  }

  CpuJob job;
  std::vector<std::thread> ts;
  uint8_t* cpu_buf = nullptr;
  size_t cpu_total = 1024ull << 20;
  if (concurrent) {
    cpu_buf = static_cast<uint8_t*>(malloc(cpu_total));
    memset(cpu_buf, 1, cpu_total);
    job.buf = cpu_buf;
    job.total = cpu_total;
    for (int t = 0; t < cpu_threads; ++t)
      ts.emplace_back(cpu_worker, t, cpu_threads, &job);
  }

  double t0 = now_s();
  double stop = t0 + duration;
  if (concurrent) {
    job.stop_time = stop;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    job.start.store(true, std::memory_order_release);
  }
  double gpu_seconds = 0;
  uint64_t gpu_bytes = 0;
  while (now_s() < stop) {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n_elems length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(gsize, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    double dt = cb.GPUEndTime - cb.GPUStartTime;
    if (dt > 0) {
      gpu_seconds += dt;
      gpu_bytes += bytes;
    }
  }
  if (concurrent) {
    for (auto& t : ts) t.join();
  }
  double t1 = now_s();

  double gpu_gbps = gpu_bytes / gpu_seconds / 1e9;
  printf("gpu_GB/s\t%.2f\n", gpu_gbps);
  if (concurrent) {
    double cpu_gbps = job.bytes_done.load() / (t1 - t0) / 1e9;
    printf("cpu_threads\t%d\n", cpu_threads);
    printf("cpu_GB/s\t%.2f\n", cpu_gbps);
    printf("total_GB/s\t%.2f\n", gpu_gbps + cpu_gbps);
    free(cpu_buf);
  }
  if (job.sink.load() == 0xDEADBEEFull) fprintf(stderr, "?");
  return 0;
}
