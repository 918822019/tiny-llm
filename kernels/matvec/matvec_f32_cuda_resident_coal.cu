// 矩阵乘向量：y = W @ x —— CUDA 权重常驻 + 合并访存 kernel 版（NVIDIA GPU）。
//
// ============================================================================
// 与 cuda_resident 的唯一区别：kernel 的访存/归约方式
// ============================================================================
//
// cuda_resident_ws 条目（docs/optimization_log.md）证伪了"alloc/free 是剩余
// 大头"的猜测，把矛头指向真正的瓶颈——**朴素 kernel 的非合并访存**：
//
//   旧 kernel：一行一个 thread。同一 warp 的 32 个线程算相邻 32 行，
//   第 i 步它们分别读 w[row][i]——地址相隔 in_dim×4 字节（896 维时
//   3584 字节）。GPU 按 128 字节 cache line 粒度取数：一次 warp 读取
//   触发 32 条分散 cache line（搬 4096 字节），真正用到的只有 128 字节
//   ——HBM 流量被放大 ~32 倍，有效带宽利用率 ~3%。A10 有 ~600 GB/s，
//   这么读法只能拿到 ~20 GB/s 上下，这就是 47 ms/tok 的来源。
//
//   新 kernel：**block-per-row**——一个 block（256 线程）合算一行。
//   线程 t 沿 in_dim 跨步取元素：t, t+256, t+512, ……同一时刻 warp 里
//   32 个线程读的是**连续 32 个 float**（128 字节，正好一条 cache line），
//   每次事务 100% 有用——这叫合并访存（coalesced access），是 GPU 编程
//   的第一定律。每个线程先攒出自己的部分和，再做 block 内归约。
//
// 归因纪律：host 包装与 cuda_resident 完全一致（权重按 host 指针常驻；
// x/y 仍是每调用分配——ws 条目已证明改它≈无效，不带进来），本步只改
// kernel 这一个变量。
//
// ============================================================================
// 归约怎么做：两级，warp 内 shuffle + warp 间 shared memory
// ============================================================================
//
// 256 线程 = 8 个 warp（每 warp 32 线程）。每个线程手里有一个部分和，
// 要汇成 1 个数：
//
//   第一级（warp 内）：__shfl_down_sync 让 lane 0 依次和 lane 16/8/4/2/1
//   的值相加——5 步后 lane 0 拿到本 warp 32 个部分和的总和。shuffle 是
//   寄存器对寄存器的交换，不经过内存，是 warp 内归约的最快路径。
//
//   第二级（warp 间）：8 个 warp 各把总和写进 shared memory 的
//   warp_sums[8]，同步后由第 0 个 warp 的 8 个 lane 读回，再做一次
//   shuffle 归约，lane 0 写出最终的 y[row]。
//
// 数值注：float 累加，且求和顺序（每线程跨步 in_dim/256 项 + 树形归约）
// 与 ref 的逐行串行不同——浮点加法不满足结合律，结果有微小舍入差，
// 单测以容差（5e-3，与 NEON 系列同口径）对齐，不要求逐位一致。

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

        constexpr int kThreads = 256;     // 每 block 线程数（= 8 个 warp）
        constexpr int kWarpSize = 32;
        constexpr int kMaxWarps = kThreads / kWarpSize;

        // ====================================================================
        // Device kernel：block-per-row + 合并访存 + 两级归约。
        // ====================================================================
        // grid-stride 以 **block** 为步长：block r 负责行 r, r+gridDim.x, ……
        // 循环变量对 block 内所有线程一致，__syncthreads() 因此安全。
        __global__ void matvec_f32_coal_kernel(const float *w, const float *x, float *y,
                                               int out_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1); // warp 内编号 0..31
            const int wid = threadIdx.x / kWarpSize;        // warp 编号 0..7

            for (int row = blockIdx.x; row < out_dim; row += gridDim.x) {
                const float *w_row = w + static_cast<size_t>(row) * in_dim;

                // ---- 阶段 1：跨步乘加（合并访存的核心）----
                // 线程 t 读 w_row[t], w_row[t+256], ……：任一时刻 warp 的
                // 32 个线程访问连续 32 个 float = 一条 128B cache line。
                // in_dim 不是 256 倍数时，尾部线程的循环自然少跑几轮；
                // in_dim < 256（如 k/v_proj 的 896 不会，但测试形状会）时，
                // 高编号线程部分和为 0，归约结果不受影响。
                float acc = 0.0f;
                for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                    acc += w_row[i] * x[i];
                }

                // ---- 阶段 2：warp 内归约（shuffle，5 步）----
                // offset 依次 16,8,4,2,1：lane L 把 lane L+offset 的值加进来，
                // 5 步后 lane 0 持有全 warp 之和。0xffffffff = 全 warp 参与。
                for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                    acc += __shfl_down_sync(0xffffffffu, acc, offset);
                }
                if (lane == 0) warp_sums[wid] = acc;
                __syncthreads(); // 等 8 个 warp 都写完 warp_sums

                // ---- 阶段 3：warp 间归约（第 0 个 warp 收尾）----
                // 只有 wid==0 的 warp 干活：前 8 个 lane 各取一个 warp 总和，
                // 其余 lane 补 0，再 shuffle 归约（kMaxWarps=8 是 2 的幂）。
                if (wid == 0) {
                    float v = (lane < kMaxWarps) ? warp_sums[lane] : 0.0f;
                    for (int offset = kMaxWarps / 2; offset > 0; offset >>= 1) {
                        v += __shfl_down_sync(0xffffffffu, v, offset);
                    }
                    if (lane == 0) y[row] = v;
                }
                __syncthreads(); // 确保下一轮复用 warp_sums 前本轮已读完
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
    } // namespace

    // ========================================================================
    // Host 包装：与 cuda_resident 完全一致（x/y 每调用分配——ws 条目已
    // 证明改它≈无效），唯一变化是启动合并访存 kernel。
    // ========================================================================
    // block-per-row：gridDim = out_dim（一行一个 block）。lm_head 的
    // 151936 行就是 15 万个 block——CUDA 调度器习以为常，不需要封顶
    // （旧 kernel 的 max_blocks=1024 封顶是因为它"一个 thread 一行"，
    // 这里并行粒度变粗了，block 数 = 行数恰好是自然映射）。
    void matvec_f32_cuda_resident_coal(const float *w, const float *x, float *y, int out_dim,
                                       int in_dim) {
        const float *d_w = resident_weight(w, out_dim, in_dim);

        const size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
        const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);
        float *d_x = nullptr, *d_y = nullptr;
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

        TINYQWEN_CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));

        const int blocks = out_dim < 1 ? 1 : out_dim; // 防御：0 行也要合法启动
        matvec_f32_coal_kernel<<<blocks, kThreads>>>(d_w, d_x, d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaDeviceSynchronize());

        TINYQWEN_CUDA_CHECK(cudaMemcpy(y, d_y, y_bytes, cudaMemcpyDeviceToHost));

        TINYQWEN_CUDA_CHECK(cudaFree(d_x));
        TINYQWEN_CUDA_CHECK(cudaFree(d_y));
    }

    // 自注册：--matvec-impl cuda_resident_coal。仅 CUDA 构建存在。
    TINYQWEN_MATVEC_VARIANT(matvec_f32_cuda_resident_coal, "cuda_resident_coal");
} // namespace tinyqwen
