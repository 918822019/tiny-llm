// 矩阵乘向量：y = W @ x —— f16 权重 + CUDA 权重常驻 + 合并访存 + 砍每调用开销版。
//
// ============================================================================
// f16 族的"满栈"：f16_cuda_resident_coal 之上再砍每调用开销
// ============================================================================
//
// 这是 f32 `cuda_resident_coal_ws` 的 f16 对应版，两个 f16 相关条目的思路合体：
//   - f16_cuda_resident_coal：权重 fp16（流量减半）+ 权重常驻 + 合并访存；
//   - cuda_resident_coal_ws：x/y 常驻 workspace（删每调用 alloc/free）+
//     删冗余 cudaDeviceSynchronize（同步 D2H 已保证 kernel 完成）。
//
// 为什么把 A（砍开销）移植到 f16：要压过 Mac 的是 f16 配置。f32 已证这一步
// 值 ~1.15×（主要来自删 sync）；f16 版预期同样收益，把 7.68 进一步压低。
//
// kernel 与 f16_cuda_resident_coal 逐字相同（读 uint16_t 权重、__half2float
// 转 fp32、合并访存 + 两级归约），本步只改 host 包装的两处开销，归因干净。
//
// 注册约定：与 f32 版**同名**（"cuda_resident_coal_ws"），按模型 dtype 解析——
// f16 模型 + `--matvec-impl cuda_resident_coal_ws` 用本文件。
//
// 数值/对齐：对齐基准仍是 f16 族的 matvec_f16_ref（权重已半精度量化），
// 容差 5e-3；删 sync 不改变数值，只改调用时序。

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
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
        // Device kernel：与 f16_cuda_resident_coal 逐字相同。
        // ====================================================================
        __global__ void matvec_f16_coal_kernel(const uint16_t *w, const float *x, float *y,
                                               int out_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1);
            const int wid = threadIdx.x / kWarpSize;
            for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
                const __half *w_row =
                        reinterpret_cast<const __half *>(w + static_cast<size_t>(row) * in_dim);
                float acc = 0.0f;
                for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                    acc += __half2float(w_row[i]) * x[i];
                }
                for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                    acc += __shfl_down_sync(0xffffffffu, acc, offset);
                }
                if (lane == 0) warp_sums[wid] = acc;
                __syncthreads();
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
        // 权重缓存：f16_cuda_resident_coal 同款（key = const uint16_t*）。
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
                return it->second.d_w;
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

        // ====================================================================
        // x/y 常驻 workspace（grow-only）：cuda_resident_coal_ws 同款。
        // ====================================================================
        struct Workspace {
            float *d_x = nullptr;
            size_t x_capacity = 0;
            float *d_y = nullptr;
            size_t y_capacity = 0;
            void ensure_x(size_t n) {
                if (n <= x_capacity) return;
                if (d_x != nullptr) TINYQWEN_CUDA_CHECK(cudaFree(d_x));
                TINYQWEN_CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
                x_capacity = n;
            }
            void ensure_y(size_t n) {
                if (n <= y_capacity) return;
                if (d_y != nullptr) TINYQWEN_CUDA_CHECK(cudaFree(d_y));
                TINYQWEN_CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
                y_capacity = n;
            }
        };
        Workspace &workspace() {
            static Workspace ws;
            return ws;
        }
    } // namespace

    // ========================================================================
    // Host 包装：f16_cuda_resident_coal 的两处开销削减版。
    // ========================================================================
    void matvec_f16_cuda_resident_coal_ws(const uint16_t *w, const float *x, float *y,
                                          int out_dim, int in_dim) {
        const uint16_t *d_w = resident_weight(w, out_dim, in_dim);

        Workspace &ws = workspace();
        ws.ensure_x(static_cast<size_t>(in_dim));
        ws.ensure_y(static_cast<size_t>(out_dim));

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(ws.d_x, x, static_cast<size_t>(in_dim) * sizeof(float),
                           cudaMemcpyHostToDevice));

        const int blocks = out_dim < 1 ? 1 : out_dim;
        matvec_f16_coal_kernel<<<blocks, kThreads>>>(d_w, ws.d_x, ws.d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        // 删冗余 sync：下面的 D2H 同步拷贝会等 kernel 完成。

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(y, ws.d_y, static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToHost));
    }

    // 自注册进 f16 注册表，名字与 f32 版共享（"cuda_resident_coal_ws"）。
    TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_cuda_resident_coal_ws, "cuda_resident_coal_ws");
} // namespace tinyqwen
