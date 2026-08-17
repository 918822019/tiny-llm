// 矩阵乘向量：y = W @ x —— CUDA 权重常驻 + x/y 常驻 workspace 版（NVIDIA GPU）。
//
// ============================================================================
// 与 cuda_resident 的唯一区别：x/y 不再每次调用 malloc/free
// ============================================================================
//
// cuda_resident 的实测（docs/optimization_log.md 的 cuda_resident 条目）：
// 47.40 ms/tok，权重常驻一步拿到 7.04×，但离 HBM 带宽地板（~3–4 ms/tok）
// 还有 ~12× 空间。剩余开销的大头是**每次调用都在做的显存分配**：
//
//   每 token ~170 次 matvec 调用 ×（cudaMalloc(x) + cudaMalloc(y)
//                                    + cudaFree(x) + cudaFree(y)）
//   ≈ 680 次 cudaMalloc/cudaFree / token
//
// cudaMalloc/cudaFree 不是免费的——驱动要维护显存分配器、可能触发同步，
// 单次几十到上百微秒，×680 就是几十毫秒级，与实测的 47 ms/tok 量级吻合。
//
// 本变体的修法：**workspace 常驻**。x 和 y 各留一块 device 缓冲区，
// 只增不减（grow-only）：
//   - 新 shape 比现有容量小 → 直接复用，零分配；
//   - 新 shape 超出容量     → 释放旧的、按新大小重分配（只在启动后
//     头几个不同 shape 上发生几次，decode 稳态下不再触发）。
// decode 里 x 恒为 hidden=896 个 float，y 最大是 lm_head 的 151936 个
// float——两块缓冲各扩张一两次就到顶，之后每调用只剩纯搬运 + 计算。
//
// 归因纪律（docs/optimization.md §6.4"一次只改一个变量"）：权重缓存逻辑
// 与 cuda_resident 相同，device kernel 仍然**逐字不变**——本步只拆
// "x/y 每调用分配"这一个变量。kernel 优化（block-per-row + 归约，吃满
// HBM 带宽）是下一层的事。
//
// 线程安全注：workspace 与权重缓存同为无锁惰性状态，前提与 cuda_resident
// 一致——decode 单流串行调用 matvec（见该文件的同名注释）。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
        // ==== fail-loud：与前两个 CUDA 变体同款 ============================
        inline void cuda_check(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen: CUDA 调用失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define TINYQWEN_CUDA_CHECK(expr) tinyqwen::cuda_check((expr), #expr)

        // ====================================================================
        // Device kernel：与 matvec_f32_cuda.cu / cuda_resident 逐字相同
        // （刻意的——本步只改 host 侧内存管理，见文件头"归因纪律"）。
        // ====================================================================
        __global__ void matvec_f32_cuda_kernel(const float *w, const float *x, float *y,
                                               int out_dim, int in_dim) {
            for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < out_dim;
                 row += gridDim.x * blockDim.x) {
                const float *w_row = w + static_cast<size_t>(row) * in_dim;
                float acc = 0.0f;
                for (int i = 0; i < in_dim; ++i) {
                    acc += w_row[i] * x[i];
                }
                y[row] = acc;
            }
        }

        // ====================================================================
        // 权重缓存：与 cuda_resident 同款（host 指针 -> device 副本）。
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
                return it->second.d_w; // 热路径
            }
            DeviceWeight &dw = cache[w];
            if (dw.d_w != nullptr) {
                TINYQWEN_CUDA_CHECK(cudaFree(dw.d_w)); // 同地址换大小的防御分支
            }
            TINYQWEN_CUDA_CHECK(cudaMalloc(&dw.d_w, bytes));
            TINYQWEN_CUDA_CHECK(cudaMemcpy(dw.d_w, w, bytes, cudaMemcpyHostToDevice));
            dw.bytes = bytes;
            return dw.d_w;
        }

        // ====================================================================
        // x/y 的常驻 workspace：grow-only 的 device 缓冲区对。
        // ====================================================================
        // 为什么 grow-only 而不是每次都精确匹配大小：
        //   精确匹配 = 每个新 shape 都要重分配，decode 里 q/k/v/o/ffn/lm_head
        //   的 shape 各不相同，会退化成"每几种 shape 各分配一次"；grow-only
        //   则让所有 shape 共享一块"到目前为止最大"的缓冲，扩张只发生
        //   在遇到更大 shape 时（整个 decode 过程屈指可数的几次）。
        //   复用大缓冲跑小 shape 完全安全——kernel 只会读写前 n 个元素。
        struct Workspace {
            float *d_x = nullptr;
            size_t x_capacity = 0; // 单位：float 个数（不是字节）
            float *d_y = nullptr;
            size_t y_capacity = 0;

            // 确保容量 >= n 个 float；不够就重分配（释放旧的）。
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
            static Workspace ws; // Meyers singleton，同 weight_cache
            return ws;
        }
    } // namespace

    // ========================================================================
    // Host 包装：与 cuda_resident 逐步对照，差异只在 x/y 的分配方式。
    // ========================================================================
    //   cuda_resident：每调用 cudaMalloc(d_x)+cudaMalloc(d_y)……末尾 cudaFree×2
    //   本 版       ：workspace().ensure_x/ensure_y（稳态零分配）……末尾不释放
    void matvec_f32_cuda_resident_ws(const float *w, const float *x, float *y, int out_dim,
                                     int in_dim) {
        const float *d_w = resident_weight(w, out_dim, in_dim);

        Workspace &ws = workspace();
        ws.ensure_x(static_cast<size_t>(in_dim));
        ws.ensure_y(static_cast<size_t>(out_dim));

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(ws.d_x, x, static_cast<size_t>(in_dim) * sizeof(float),
                           cudaMemcpyHostToDevice));

        const int threads_per_block = 256;
        const int max_blocks = 1024;
        int blocks = (out_dim + threads_per_block - 1) / threads_per_block;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        matvec_f32_cuda_kernel<<<blocks, threads_per_block>>>(d_w, ws.d_x, ws.d_y, out_dim,
                                                              in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaDeviceSynchronize());

        TINYQWEN_CUDA_CHECK(
                cudaMemcpy(y, ws.d_y, static_cast<size_t>(out_dim) * sizeof(float),
                           cudaMemcpyDeviceToHost));
        // 注意：这里刻意不释放 ws.d_x / ws.d_y——常驻正是本变体的全部意图。
    }

    // 自注册：--matvec-impl cuda_resident_ws。仅 CUDA 构建存在。
    TINYQWEN_MATVEC_VARIANT(matvec_f32_cuda_resident_ws, "cuda_resident_ws");
} // namespace tinyqwen
