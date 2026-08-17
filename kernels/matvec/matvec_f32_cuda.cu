// 矩阵乘向量：y = W @ x —— CUDA 参考版（NVIDIA GPU）。
//
// ============================================================================
// 为什么叫"参考版"，不是"优化版"
// ============================================================================
//
// 这台机器（x86_64 + A10）之前完全没有任何加速路径：所有 NEON 变体都包在
// `#if defined(__aarch64__)` 里，在这里统统编译成空文件，dispatch 兜底回
// 标量 `matvec_f32_ref`。这个文件的目标不是"比 CPU 快"，而是先把
// "CUDA kernel 怎么接进现有 dispatch"这条路走通，量出第一个真实数字：
//   - 正确性优先：kernel 本身故意写得直白（一行一个 thread，串行点积），
//     不做 shared memory / warp reduction；
//   - Host 侧包装故意"笨"：每次调用都重新 cudaMalloc + H2D 拷贝 + kernel
//     启动 + D2H 拷贝 + cudaFree。这对 decode 这种"每 token 调用上百次
//     matvec、每次张量才几 KB~几 MB"的场景是反模式——真实开销主要来自
//     PCIe 传输延迟和 kernel 启动延迟，不是计算本身。这里刻意不隐藏这个
//     代价，用 `scripts/record.sh` 测出来的就是这个代价的真实大小。
//   - 下一步该往哪优化（等看到实测数字再决定要不要做）：
//       ① 权重常驻显存——权重在整个 decode 期间地址不变，没必要每次调用
//         都重新上传，只需要按 host 指针缓存一份 device 副本；
//       ② block-per-row + shared memory 归约——现在一行一个 thread 对
//         大 in_dim（几千）是串行的，并行度全靠"行数"（out_dim），
//         小矩阵（k_proj/v_proj，out_dim 仅 128）根本喂不满 A10 的 SM；
//       ③ 更进一步是把整段 forward 常驻显存，省掉每个算子后都要经过
//         host 的中转——但那是"要不要把整个 runtime 搬上 GPU"的架构级
//         决策，不是这个 kernel 该管的事。
//
// ============================================================================
// GPU 编程模型的最小直觉（读下面代码前先建立）
// ============================================================================
//
// CUDA 把计算切成 grid（网格）→ block（线程块）→ thread（线程）三层：
//   - 你写一个 kernel 函数（下面的 matvec_f32_cuda_kernel），描述"一个
//     thread 该干什么活"；
//   - 启动时指定 grid 里有几个 block、每个 block 里有几个 thread，GPU
//     把这些 thread 分发到硬件的流处理器上**同时**跑；
//   - 每个 thread 通过 `blockIdx`/`threadIdx`/`blockDim` 算出"我是谁"，
//     决定自己该处理哪份数据——这里是"我该算 y 的第几行"。
// 这和 CPU 多线程池（见 `matvec_f32_neon_mt.cpp`）的思路是同一件事
// （把总任务切给多个执行单元），只是 GPU 的执行单元数量级远大于 CPU
// 核数（A10 有 72 个 SM，每个 SM 能同时驻留上千线程）。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
        // ==== fail-loud：CUDA API 调用出错直接打印原因 + abort ==========
        // 与 dispatch.cpp 里"ref 未注册就 abort"、matvec_f32_neon_mt 里
        // 线程池同步失败即崩的风格一致——本仓库的约定是宁可崩溃到位，
        // 不悄悄兜底出错误数字。
        inline void cuda_check(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen: CUDA 调用失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define TINYQWEN_CUDA_CHECK(expr) tinyqwen::cuda_check((expr), #expr)

        // ====================================================================
        // Device kernel：一个 thread 负责 y 的一行。
        // ====================================================================
        //
        // grid-stride loop：thread 总数可能少于 out_dim（比如 out_dim 很大时
        // 我们不会真的起那么多 block），所以每个 thread 按
        // `row += gridDim.x * blockDim.x`（一次"跨步"处理的行数）循环处理
        // 多行，直到把 [0, out_dim) 都覆盖完——这是 CUDA 里让 kernel 与
        // "启动了多少个 thread"解耦的标准写法，即使启动配置算少了也不会
        // 漏算，只是并行度打折。
        //
        // 单行内部：和 `matvec_f32_ref` 数学上做的是同一件事——第 row 行
        // 与 x 逐元素相乘再求和——只是这里用 float 累加（不像 ref 用
        // double），精度取舍与 NEON 系列变体一致，误差走单测容差门禁。
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
    } // namespace

    // ========================================================================
    // Host 包装：分发层调用的入口，签名与 MatvecFn 完全一致。
    // ========================================================================
    //
    // 步骤（每次调用都完整走一遍，见文件头"为什么叫参考版"）：
    //   1. 在显存上开三块缓冲区（W、x、y）；
    //   2. 把 W、x 从 host 内存拷进显存（H2D = host to device）；
    //   3. 启动 kernel，等它跑完；
    //   4. 把结果 y 从显存拷回 host 内存（D2H）；
    //   5. 释放显存。
    // 2/4 两步走 PCIe，是这个"参考版"最大的隐藏成本来源。
    void matvec_f32_cuda(const float *w, const float *x, float *y, int out_dim, int in_dim) {
        const size_t w_bytes = static_cast<size_t>(out_dim) * in_dim * sizeof(float);
        const size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
        const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);

        float *d_w = nullptr, *d_x = nullptr, *d_y = nullptr;
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_w, w_bytes));
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

        TINYQWEN_CUDA_CHECK(cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));

        // 256 个 thread 一个 block（常见的通用取值）；block 数按 out_dim
        // 铺开，但封顶——out_dim 很大（比如 lm_head 的词表维度）时不需要
        // 真起几千个 block，grid-stride loop 会让少量 block 循环覆盖全部行。
        const int threads_per_block = 256;
        const int max_blocks = 1024;
        int blocks = (out_dim + threads_per_block - 1) / threads_per_block;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        matvec_f32_cuda_kernel<<<blocks, threads_per_block>>>(d_w, d_x, d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError()); // 捕获启动本身的错误（如非法配置）
        TINYQWEN_CUDA_CHECK(cudaDeviceSynchronize()); // 等 kernel 跑完，顺带捕获执行期错误

        TINYQWEN_CUDA_CHECK(cudaMemcpy(y, d_y, y_bytes, cudaMemcpyDeviceToHost));

        TINYQWEN_CUDA_CHECK(cudaFree(d_w));
        TINYQWEN_CUDA_CHECK(cudaFree(d_x));
        TINYQWEN_CUDA_CHECK(cudaFree(d_y));
    }

    // 自注册进 dispatch：--matvec-impl cuda / tinyqwen.conf 里
    // matvec_impl = cuda 即可选用。只有检测到 nvcc 的构建才会编译到这里
    // （见 kernels/CMakeLists.txt），其他平台的可用列表里不会出现 "cuda"。
    TINYQWEN_MATVEC_VARIANT(matvec_f32_cuda, "cuda");
} // namespace tinyqwen
