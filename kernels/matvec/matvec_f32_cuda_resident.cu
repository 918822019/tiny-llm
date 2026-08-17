// 矩阵乘向量：y = W @ x —— CUDA 权重常驻版（NVIDIA GPU）。
//
// ============================================================================
// 与 matvec_f32_cuda.cu（参考版）的唯一区别：权重不再每次调用重传
// ============================================================================
//
// 参考版的实测教训（docs/optimization_log.md 的 cuda 条目）：333.75 ms/tok，
// 输给本机最好的 CPU 标量变体 acc4（269 ms/tok）。原因不是 GPU 算力，
// 是"传输税"——每次调用都把整块权重经 PCIe 重新上传（decode 每 token
// ~2GB 流量，170 次 matvec 调用），PCIe 成了主瓶颈，A10 的 HBM 带宽
// （~600 GB/s）根本没派上用场。
//
// 本变体只做一件事：**权重常驻显存**。decode 期间权重内容永远不变，
// 那就第一次见到某块权重时上传一次、记住它的 device 副本，之后每次
// 调用直接用缓存——每 token 的 PCIe 上行流量从 ~2GB 掉到 x 的几 KB。
//
// 归因纪律（docs/optimization.md §6.4"一次只改一个变量"）：device kernel
// 与参考版**逐字相同**（一行一个 thread、行内串行点积、无 shared memory /
// warp 归约）。这样测出来的提升可以完整归给"权重常驻"这一件事；kernel
// 侧优化（block-per-row + 归约）是下一层的事。
//
// ============================================================================
// 缓存设计：按 host 指针做 key
// ============================================================================
//
//   host 权重指针 w  ──查表──>  { device 指针 d_w, 字节数 bytes }
//
//   - 表里没有 w          → cudaMalloc + 一次性 H2D 上传，记入表；
//   - 表里有 w 且字节匹配 → 直接复用 d_w（这是 decode 稳态下的热路径）；
//   - 表里有 w 但字节不符 → 防御性分支：释放旧副本、重新上传
//     （同一地址被复用来放别的矩阵时不至于算错，见下面的安全前提）。
//
// 为什么按 host 指针做 key 在这里是安全的：
//   runtime 的 loader 一次性 fread 整个 .tqwen 进内存，所有权重 tensor
//   是指向这块内存的视图（runtime/model_loader.cpp），进程存活期间
//   地址、内容都不变——"指针相同 ⇒ 内容相同"成立。这个前提写在
//   docs/known_limitations.md 的边界里；哪天 loader 换成 mmap 逐页换入
//   或分块流式加载，这个 key 就不再可靠，必须换成显式的权重生命周期
//   管理（加载时统一上传、卸载时释放）。
//
// 容量注：缓存只增不减，上限 = 整个模型的权重大小（0.5B fp32 ≈ 1.98GB，
// A10 的 24GB 显存绰绰有余）。换更大的模型前需要重新算这笔账；多模型
// 共存则需要 LRU 或按模型显式管理——v1 范围外，先不做。
//
// 线程安全注：缓存是惰性构建的普通 unordered_map，无锁。本 runtime 的
// decode 是单流串行的（matvec 调用不会并发），与 matvec_f32_neon_mt 的
// 线程池不同——那里并行的是 kernel 内部，dispatch 调用本身仍是串行的。
// 若将来引入多流/多 batch 并发调用 matvec，这里要加锁或改成每流一份。

#include "dispatch.h" // TINYQWEN_MATVEC_VARIANT 自注册宏
#include "ref_ops.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
        // ==== fail-loud：与参考版同款，CUDA API 出错打印原因 + abort ======
        inline void cuda_check(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen: CUDA 调用失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define TINYQWEN_CUDA_CHECK(expr) tinyqwen::cuda_check((expr), #expr)

        // ====================================================================
        // Device kernel：与 matvec_f32_cuda.cu 逐字相同（刻意的，见文件头
        // "归因纪律"）。一行一个 thread + grid-stride，行内串行 float 点积。
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
        // 权重缓存：host 指针 -> device 副本。
        // ====================================================================
        // Meyers singleton：首次调用时构造，C++11 起线程安全，与
        // dispatch.cpp 的注册表同款模式（无静态初始化顺序坑）。
        struct DeviceWeight {
            float *d_w = nullptr; // 显存副本
            size_t bytes = 0;     // 副本大小（字节），用于防御性校验
        };

        std::unordered_map<const float *, DeviceWeight> &weight_cache() {
            static std::unordered_map<const float *, DeviceWeight> c;
            return c;
        }

        // 查缓存；未命中（或字节不符）就上传并（重新）登记。
        // 返回可直接传给 kernel 的 device 指针。
        const float *resident_weight(const float *w, int out_dim, int in_dim) {
            const size_t bytes = static_cast<size_t>(out_dim) * in_dim * sizeof(float);
            auto &cache = weight_cache();
            auto it = cache.find(w);
            if (it != cache.end() && it->second.bytes == bytes) {
                return it->second.d_w; // 热路径：decode 稳态下每次都走这里
            }
            // 冷路径：首次见到这块权重（或同地址换了不同大小的矩阵）。
            DeviceWeight &dw = cache[w];
            if (dw.d_w != nullptr) {
                TINYQWEN_CUDA_CHECK(cudaFree(dw.d_w)); // 字节不符的防御分支
            }
            TINYQWEN_CUDA_CHECK(cudaMalloc(&dw.d_w, bytes));
            TINYQWEN_CUDA_CHECK(cudaMemcpy(dw.d_w, w, bytes, cudaMemcpyHostToDevice));
            dw.bytes = bytes;
            return dw.d_w;
        }
    } // namespace

    // ========================================================================
    // Host 包装：与参考版逐步对照，差异只有第 1 步。
    // ========================================================================
    //   参考版：cudaMalloc(W) + H2D(W)  ……每调用一次都重来一遍
    //   本 版：resident_weight(W)        ……首次上传，之后命中缓存零拷贝
    // x/y 仍是每次调用分配 + 搬运（只有几 KB，不是瓶颈；下一层再优化）。
    void matvec_f32_cuda_resident(const float *w, const float *x, float *y, int out_dim,
                                  int in_dim) {
        const size_t x_bytes = static_cast<size_t>(in_dim) * sizeof(float);
        const size_t y_bytes = static_cast<size_t>(out_dim) * sizeof(float);

        // 第 1 步（唯一的变化）：取权重的常驻 device 副本。
        const float *d_w = resident_weight(w, out_dim, in_dim);

        float *d_x = nullptr, *d_y = nullptr;
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
        TINYQWEN_CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

        TINYQWEN_CUDA_CHECK(cudaMemcpy(d_x, x, x_bytes, cudaMemcpyHostToDevice));

        const int threads_per_block = 256;
        const int max_blocks = 1024;
        int blocks = (out_dim + threads_per_block - 1) / threads_per_block;
        if (blocks > max_blocks) blocks = max_blocks;
        if (blocks < 1) blocks = 1;

        matvec_f32_cuda_kernel<<<blocks, threads_per_block>>>(d_w, d_x, d_y, out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaDeviceSynchronize());

        TINYQWEN_CUDA_CHECK(cudaMemcpy(y, d_y, y_bytes, cudaMemcpyDeviceToHost));

        TINYQWEN_CUDA_CHECK(cudaFree(d_x));
        TINYQWEN_CUDA_CHECK(cudaFree(d_y));
    }

    // 自注册：--matvec-impl cuda_resident / tinyqwen.conf 里
    // matvec_impl = cuda_resident。仅 CUDA 构建存在（见 kernels/CMakeLists.txt）。
    TINYQWEN_MATVEC_VARIANT(matvec_f32_cuda_resident, "cuda_resident");
} // namespace tinyqwen
