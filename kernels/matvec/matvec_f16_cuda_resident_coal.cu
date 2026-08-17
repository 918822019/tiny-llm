// 矩阵乘向量：y = W @ x —— f16 权重 + CUDA 权重常驻 + 合并访存版（NVIDIA GPU）。
//
// ============================================================================
// 在 f32 cuda_resident_coal 之上，唯一的变化：权重精度 fp32 → fp16
// ============================================================================
//
// f32 cuda_resident_coal（docs/optimization_log.md 同名条目）把 decode 打到
// 9.16 ms/tok，离 f32 的 HBM 带宽地板（~3–4 ms/tok）只剩固定开销。但 decode
// 是纯"搬权重"负载，想再往下走，最直接的一刀就是**减少搬运**——把权重从
// fp32（4 字节/元素）换成 fp16（2 字节/元素），每 token 的 HBM 流量直接
// 减半（本模型 1.98GB → 0.99GB），带宽地板随之从 ~3.3ms 降到 ~1.7ms。
//
// 这与 CPU 侧 fp16 路线（f16_neon_mt_kv_nt，流量减半带来 1.75×）是同一规律，
// 也是"归因四分类"里的**减少搬运**类。weight-only 半精度：只省权重的存储/
// 搬运，x/y 与累加仍是 fp32，计算精度不降（与 f16_ref 的契约一致）。
//
// 归因纪律：本变体相对 f32 cuda_resident_coal 只改"权重 dtype"这一个变量——
// host 包装（权重常驻 + 每调用 x/y 分配）与合并访存 kernel 的归约结构全部
// 照搬，唯一差别是 kernel 读 `uint16_t` 权重并用 `__half2float` 转成 fp32
// 再乘加。这样测出的提升可以干净地归给"fp16 减半流量"。
//
// 注册约定（与 CPU 侧 f16 族一致）：实现名与 f32 注册表**共享命名空间**，
// 都叫 "cuda_resident_coal"。main 按模型文件的 dtype 决定查哪张表——
// f32 模型 + `--matvec-impl cuda_resident_coal` 走 f32 版，f16 模型走本版。
// 见 dispatch.h 的 f16 路径注释。
//
// 数值注：fp16→fp32 转换无损（half 是 float 的子集），累加用 fp32，求和顺序
// 与 f32 coal 版相同，因此本版与 **matvec_f16_ref**（double 累加）的误差，
// 同 f32 coal 与 f32 ref 的误差一个量级（容差 5e-3）。注意对齐基准是
// f16 族的 matvec_f16_ref，不是 f32 的——权重已经半精度量化，和 fp32 权重
// 的 ref 本就逐位不同，不能跨族对齐。

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <cuda_fp16.h>   // __half / __half2float
#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
        // ==== fail-loud：与其他 CUDA 变体同款 ==============================
        inline void cuda_check(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen: CUDA 调用失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define TINYQWEN_CUDA_CHECK(expr) tinyqwen::cuda_check((expr), #expr)

        constexpr int kThreads = 256;
        constexpr int kWarpSize = 32;
        constexpr int kMaxWarps = kThreads / kWarpSize;

        // ====================================================================
        // Device kernel：结构与 f32 cuda_resident_coal 完全一致，唯一差别是
        // 权重按 uint16_t（fp16）加载、转 fp32 再乘加。
        // ====================================================================
        __global__ void matvec_f16_coal_kernel(const uint16_t *w, const float *x, float *y,
                                               int out_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1);
            const int wid = threadIdx.x / kWarpSize;

            for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
                // fp16 权重行。用 __half* 视图加载，__half2float 是原生
                // F16→F32 转换指令（比 ref_ops.h 的位操作版快得多）。
                const __half *w_row =
                        reinterpret_cast<const __half *>(w + static_cast<size_t>(row) * in_dim);

                // ---- 阶段 1：跨步乘加（合并访存）----
                // 线程 t 读 w_row[t], w_row[t+256], ……：warp 内任一时刻读
                // 连续 32 个 fp16 = 64 字节（半条 cache line）。比 f32 版每条
                // 指令搬的字节少一半——这正是流量减半的来源；字节总数也减半，
                // 净效果是 HBM 读带宽压力 ÷2。
                float acc = 0.0f;
                for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                    acc += __half2float(w_row[i]) * x[i];
                }

                // ---- 阶段 2：warp 内归约（与 f32 版逐字相同）----
                for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                    acc += __shfl_down_sync(0xffffffffu, acc, offset);
                }
                if (lane == 0) warp_sums[wid] = acc;
                __syncthreads();

                // ---- 阶段 3：warp 间归约 ----
                if (wid == 0) {
                    float v = (lane < kMaxWarps) ? warp_sums[lane] : 0.0f;
                    for (int offset = kMaxWarps / 2; offset > 0; offset >>= 1) {
                        v += __shfl_down_sync(0xffffffffu, v, offset);
                    }
                    if (lane == 0) y[row] = v;
                }
                __syncthreads();
            }
        }

        // ====================================================================
        // 权重缓存：与 f32 coal 同款，key 换成 const uint16_t*（f16 权重指针），
        // 字节数按 sizeof(uint16_t) 计。本文件是独立 TU，与 f32 版的缓存互不干扰。
        // ====================================================================
        struct DeviceWeight {
            uint16_t *d_w = nullptr;
            size_t bytes = 0;
        };

        std::unordered_map<const uint16_t *, DeviceWeight> &weight_cache() {
            static std::unordered_map<const uint16_t *, DeviceWeight> c;
            return c;
        }

        const uint16_t *resident_weight(const uint16_t *w, int out_dim, int in_dim) {
            const size_t bytes = static_cast<size_t>(out_dim) * in_dim * sizeof(uint16_t);
            auto &cache = weight_cache();
            auto it = cache.find(w);
            if (it != cache.end() && it->second.bytes == bytes) {
                return it->second.d_w; // 热路径
            }
            DeviceWeight &dw = cache[w];
            if (dw.d_w != nullptr) {
                TINYQWEN_CUDA_CHECK(cudaFree(dw.d_w));
            }
            TINYQWEN_CUDA_CHECK(cudaMalloc(&dw.d_w, bytes));
            TINYQWEN_CUDA_CHECK(cudaMemcpy(dw.d_w, w, bytes, cudaMemcpyHostToDevice));
            dw.bytes = bytes;
            return dw.d_w;
        }
    } // namespace

    // ========================================================================
    // Host 包装：与 f32 cuda_resident_coal 逐步一致（权重常驻 + 每调用 x/y
    // 分配 + 启动合并访存 kernel + 同步回读），只是权重类型/字节数换成 f16。
    // ========================================================================
    void matvec_f16_cuda_resident_coal(const uint16_t *w, const float *x, float *y,
                                       int out_dim, int in_dim) {
        const uint16_t *d_w = resident_weight(w, out_dim, in_dim);

        const size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
        const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);
        float *d_x = nullptr, *d_y = nullptr;
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

        TINYQWEN_CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));

        const int blocks = out_dim < 1 ? 1 : out_dim;
        matvec_f16_coal_kernel<<<blocks, kThreads>>>(d_w, d_x, d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaDeviceSynchronize());

        TINYQWEN_CUDA_CHECK(cudaMemcpy(y, d_y, y_bytes, cudaMemcpyDeviceToHost));

        TINYQWEN_CUDA_CHECK(cudaFree(d_x));
        TINYQWEN_CUDA_CHECK(cudaFree(d_y));
    }

    // 自注册进 **f16 注册表**，名字与 f32 版共享（"cuda_resident_coal"）。
    // 仅 CUDA 构建存在；f16 模型 + --matvec-impl cuda_resident_coal 即用本版。
    TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_cuda_resident_coal, "cuda_resident_coal");
} // namespace tinyqwen
