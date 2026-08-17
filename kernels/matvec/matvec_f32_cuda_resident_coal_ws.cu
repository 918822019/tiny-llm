// 矩阵乘向量：y = W @ x —— CUDA 权重常驻 + 合并访存 + 砍每调用开销版。
//
// ============================================================================
// 在 cuda_resident_coal 之上，只改 host 包装的两处开销，kernel 逐字不变
// ============================================================================
//
// f16 条目（docs/optimization_log.md）的 Amdahl 诊断：decode 时间大头已从
// "搬权重"转移到"每调用的固定开销"。cuda_resident_coal 每生成一个 token
// 要做 ~170 次 matvec，每次的 host 包装是：
//
//   cudaMalloc(d_x) + cudaMalloc(d_y)      ← 2 次显存分配
//   cudaMemcpy H2D (x)
//   kernel 启动
//   cudaDeviceSynchronize()                ← 显式全设备同步
//   cudaMemcpy D2H (y)
//   cudaFree(d_x) + cudaFree(d_y)          ← 2 次显存释放
//
// 本变体砍掉其中两处：
//
//   ① x/y 改 **grow-only 常驻 workspace**——删掉每调用 4 次 alloc/free
//      （~680 次/token）。复用 cuda_resident_ws 的 workspace 思路；那次在
//      kernel 还占大头时测得≈0，如今 kernel 已降到 ~4ms，固定开销成了主矛盾，
//      这一刀才真正值钱（详见 ws 条目的"时序性"教训）。
//
//   ② **删掉冗余的 cudaDeviceSynchronize()**。它之所以能删：紧随其后的
//      D2H `cudaMemcpy` 本身就是**同步**的——它排在 kernel 之后的默认流上，
//      会等 kernel 跑完才把数据拷回并返回。也就是说 host 包装返回时 kernel
//      必然已完成，显式 sync 是重复保险，只留下每次 ~数微秒的调用开销。
//      删掉它不影响正确性，也不引入竞态：
//        - workspace 复用：上一调用的 D2H 已同步完成 → 其 kernel 读完 d_x、
//          写完 d_y 都发生在返回之前，下一调用覆写 d_x/d_y 安全；
//        - 权重缓存扩容/重传（冷路径）同理发生在上一调用完全结束之后。
//
// 保留不动的：kernel 本体（与 cuda_resident_coal 逐字相同）、每调用一次的
// kernel 启动、H2D x / D2H y 两次拷贝。启动次数与拷贝次数要靠下一步的
// qkv/gate_up 融合（减少调用数）或 CUDA Graph 才能降——不在本步范围。
//
// 归因注：本步同时改了 alloc/free（ws 已单独证明≈0）与 sync 两处，属"砍
// 每调用开销"这一个主题下的组合改动；实测增量主要归给 sync 删除 + 与
// workspace 的交互，日志里如实拆分说明。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

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
        // Device kernel：与 cuda_resident_coal 逐字相同（本步只改 host 包装）。
        // ====================================================================
        __global__ void matvec_f32_coal_kernel(const float *w, const float *x, float *y,
                                               int out_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1);
            const int wid = threadIdx.x / kWarpSize;
            for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
                const float *w_row = w + static_cast<size_t>(row) * in_dim;
                float acc = 0.0f;
                for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                    acc += w_row[i] * x[i];
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
        // 权重缓存：与 cuda_resident_coal 同款（host 指针 -> device 副本）。
        // ====================================================================
        struct DeviceWeight {
            float *d_w = nullptr;
            size_t bytes = 0;
        };
        std::unordered_map<const float *, DeviceWeight> &weight_cache() {
            static std::unordered_map<const float *, DeviceWeight> c;
            return c;
        }
        const float *resident_weight(const float *w, int out_dim, int in_dim) {
            const size_t bytes = static_cast<size_t>(out_dim) * in_dim * sizeof(float);
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
        // x/y 常驻 workspace：与 cuda_resident_ws 同款（grow-only）。
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
    // Host 包装：cuda_resident_coal 的两处开销削减版。
    // ========================================================================
    void matvec_f32_cuda_resident_coal_ws(const float *w, const float *x, float *y,
                                          int out_dim, int in_dim) {
        const float *d_w = resident_weight(w, out_dim, in_dim);

        // ① x/y 用常驻 workspace，替代每调用 cudaMalloc/cudaFree。
        Workspace &ws = workspace();
        ws.ensure_x(static_cast<size_t>(in_dim));
        ws.ensure_y(static_cast<size_t>(out_dim));

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(ws.d_x, x, static_cast<size_t>(in_dim) * sizeof(float),
                           cudaMemcpyHostToDevice));

        const int blocks = out_dim < 1 ? 1 : out_dim;
        matvec_f32_coal_kernel<<<blocks, kThreads>>>(d_w, ws.d_x, ws.d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        // ② 不再 cudaDeviceSynchronize：下面的 D2H 是同步的，会等 kernel 完成。

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(y, ws.d_y, static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToHost));
        // 不释放 ws.d_x / ws.d_y（常驻）。
    }

    // 自注册：--matvec-impl cuda_resident_coal_ws。仅 CUDA 构建存在。
    TINYQWEN_MATVEC_VARIANT(matvec_f32_cuda_resident_coal_ws, "cuda_resident_coal_ws");
} // namespace tinyqwen
