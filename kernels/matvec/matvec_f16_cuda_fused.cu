// f16 CUDA 融合 matvec：qkv 三合一 + gate/up（pair）二合一。
//
// ============================================================================
// 为什么要融合：把"多次 matvec 调用"合成"一次"，砍掉每调用的固定开销
// ============================================================================
//
// cuda_resident_coal_ws 条目已诊断：逐调用优化触底后，剩余开销是"每调用
// 必付的三件套"——kernel 启动 + H2D x + D2H y（各 ~170 次/token）。要再降，
// 只能**减少调用次数**。decode 里天然有共享输入 x 的 matvec 组：
//   - qkv：q_proj、k_proj、v_proj 都乘同一个 normed 向量；
//   - gate_up：gate_proj、up_proj 都乘同一个 normed 向量。
// CPU 侧早就用 fork-join 把它们合并（见 matvec_qkv / matvec_pair 入口）；
// GPU 侧之前没注册对应实现，dispatch 兜底成多次单独 matvec——本文件补上。
//
// 融合后每组从"N 次调用"变"1 次调用"：
//   qkv（3 个矩阵）：1 次 H2D x + 1 个 kernel + 3 次 D2H（原来 3 H2D+3 kernel+3 D2H）
//   pair（2 个矩阵）：1 次 H2D x + 1 个 kernel + 2 次 D2H（原来 2 H2D+2 kernel+2 D2H）
// 每层 matvec 调用从 7 次降到 4 次，启动与 x 上传次数近乎减半。
//
// 实现要点：一个 kernel 里让 block 按"全局行号"认领输出行——
//   pair：block r ∈ [0, 2*out_dim)，r<out_dim 算 W1 的第 r 行写到 y1，
//         否则算 W2 的第 (r-out_dim) 行写到 y2；
//   qkv ：block r 依次落在 q / k / v 三段，各取对应权重与输出。
// 每行内部仍是合并访存 + 两级归约（与 coal 系列逐字同款），x 常驻、权重
// 常驻、删冗余 sync 全部沿用。数值与"分开调 matvec_f16"一致（同一套浮点
// 顺序，每行独立），对齐基准仍是 matvec_f16_ref。
//
// 注册约定：qkv / pair 各有独立注册表（dispatch.h），实现名与 matvec 主表
// 共享 "cuda_resident_coal_ws"——model 的 mv_qkv / mv_pair 按当前 f16 impl
// 名查表命中本实现；未注册的算子（单 matvec）仍走 matvec_f16_cuda_resident_coal_ws。

#include "dispatch.h"
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
        // pair kernel：y1 = W1@x，y2 = W2@x。gridDim = 2*out_dim，一个 block
        // 一行；前半 block 用 W1/y1，后半用 W2/y2。
        // ====================================================================
        __global__ void f16_pair_coal_kernel(const uint16_t *w1, const uint16_t *w2,
                                             const float *x, float *y1, float *y2,
                                             int out_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1);
            const int wid = threadIdx.x / kWarpSize;
            const int grow = blockIdx.x; // 全局行号 ∈ [0, 2*out_dim)

            const uint16_t *W;
            float *out;
            int row;
            if (grow < out_dim) { W = w1; out = y1; row = grow; }
            else { W = w2; out = y2; row = grow - out_dim; }

            const __half *w_row =
                    reinterpret_cast<const __half *>(W + static_cast<size_t>(row) * in_dim);
            float acc = 0.0f;
            for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                acc += __half2float(w_row[i]) * x[i];
            }
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, offset);
            if (lane == 0) warp_sums[wid] = acc;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < kMaxWarps) ? warp_sums[lane] : 0.0f;
                for (int offset = kMaxWarps / 2; offset > 0; offset >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                if (lane == 0) out[row] = v;
            }
        }

        // ====================================================================
        // qkv kernel：yq = Wq@x（q_dim 行）、yk = Wk@x、yv = Wv@x（各 kv_dim 行）。
        // gridDim = q_dim + 2*kv_dim；block 按全局行号落入 q / k / v 三段。
        // ====================================================================
        __global__ void f16_qkv_coal_kernel(const uint16_t *wq, const uint16_t *wk,
                                            const uint16_t *wv, const float *x,
                                            float *yq, float *yk, float *yv,
                                            int q_dim, int kv_dim, int in_dim) {
            __shared__ float warp_sums[kMaxWarps];
            const int lane = threadIdx.x & (kWarpSize - 1);
            const int wid = threadIdx.x / kWarpSize;
            const int grow = blockIdx.x; // ∈ [0, q_dim + 2*kv_dim)

            const uint16_t *W;
            float *out;
            int row;
            if (grow < q_dim) {
                W = wq; out = yq; row = grow;
            } else if (grow < q_dim + kv_dim) {
                W = wk; out = yk; row = grow - q_dim;
            } else {
                W = wv; out = yv; row = grow - q_dim - kv_dim;
            }

            const __half *w_row =
                    reinterpret_cast<const __half *>(W + static_cast<size_t>(row) * in_dim);
            float acc = 0.0f;
            for (int i = threadIdx.x; i < in_dim; i += kThreads) {
                acc += __half2float(w_row[i]) * x[i];
            }
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
                acc += __shfl_down_sync(0xffffffffu, acc, offset);
            if (lane == 0) warp_sums[wid] = acc;
            __syncthreads();
            if (wid == 0) {
                float v = (lane < kMaxWarps) ? warp_sums[lane] : 0.0f;
                for (int offset = kMaxWarps / 2; offset > 0; offset >>= 1)
                    v += __shfl_down_sync(0xffffffffu, v, offset);
                if (lane == 0) out[row] = v;
            }
        }

        // ====================================================================
        // 权重缓存：与其他 f16 CUDA 变体同款（key = const uint16_t*）。
        // ====================================================================
        struct DeviceWeight {
            uint16_t *d_w = nullptr;
            size_t bytes = 0;
        };
        std::unordered_map<const uint16_t *, DeviceWeight> &weight_cache() {
            static std::unordered_map<const uint16_t *, DeviceWeight> c;
            return c;
        }
        const uint16_t *resident_weight(const uint16_t *w, size_t elems) {
            const size_t bytes = elems * sizeof(uint16_t);
            auto &cache = weight_cache();
            auto it = cache.find(w);
            if (it != cache.end() && it->second.bytes == bytes) return it->second.d_w;
            DeviceWeight &dw = cache[w];
            if (dw.d_w != nullptr) TINYQWEN_CUDA_CHECK(cudaFree(dw.d_w));
            TINYQWEN_CUDA_CHECK(cudaMalloc(&dw.d_w, bytes));
            TINYQWEN_CUDA_CHECK(cudaMemcpy(dw.d_w, w, bytes, cudaMemcpyHostToDevice));
            dw.bytes = bytes;
            return dw.d_w;
        }

        // ====================================================================
        // 输出 workspace（grow-only）。pair 用 d_y1/d_y2；qkv 用 d_yq/d_yk/d_yv。
        // x 用 d_x。全部常驻，删每调用 alloc/free。
        // ====================================================================
        struct FusedWs {
            float *d_x = nullptr; size_t x_cap = 0;
            float *d_y1 = nullptr; size_t y1_cap = 0;
            float *d_y2 = nullptr; size_t y2_cap = 0;
            float *d_yq = nullptr; size_t yq_cap = 0;
            float *d_yk = nullptr; size_t yk_cap = 0;
            float *d_yv = nullptr; size_t yv_cap = 0;
            static void grow(float *&p, size_t &cap, size_t n) {
                if (n <= cap) return;
                if (p != nullptr) TINYQWEN_CUDA_CHECK(cudaFree(p));
                TINYQWEN_CUDA_CHECK(cudaMalloc(&p, n * sizeof(float)));
                cap = n;
            }
        };
        FusedWs &ws() {
            static FusedWs w;
            return w;
        }
    } // namespace

    // ========================================================================
    // pair host 包装：1 H2D x + 1 kernel + 2 D2H。
    // ========================================================================
    void matvec_f16_pair_fused(const uint16_t *w1, const uint16_t *w2, const float *x,
                               float *y1, float *y2, int out_dim, int in_dim) {
        const uint16_t *d_w1 = resident_weight(w1, static_cast<size_t>(out_dim) * in_dim);
        const uint16_t *d_w2 = resident_weight(w2, static_cast<size_t>(out_dim) * in_dim);
        FusedWs &W = ws();
        FusedWs::grow(W.d_x, W.x_cap, static_cast<size_t>(in_dim));
        FusedWs::grow(W.d_y1, W.y1_cap, static_cast<size_t>(out_dim));
        FusedWs::grow(W.d_y2, W.y2_cap, static_cast<size_t>(out_dim));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(W.d_x, x, static_cast<size_t>(in_dim) * sizeof(float),
                                       cudaMemcpyHostToDevice));
        const int blocks = 2 * out_dim;
        f16_pair_coal_kernel<<<blocks, kThreads>>>(d_w1, d_w2, W.d_x, W.d_y1, W.d_y2,
                                                   out_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaMemcpy(y1, W.d_y1, static_cast<size_t>(out_dim) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(y2, W.d_y2, static_cast<size_t>(out_dim) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
    }

    // ========================================================================
    // qkv host 包装：1 H2D x + 1 kernel + 3 D2H。
    // ========================================================================
    void matvec_f16_qkv_fused(const uint16_t *wq, const uint16_t *wk, const uint16_t *wv,
                              const float *x, float *yq, float *yk, float *yv,
                              int q_dim, int kv_dim, int in_dim) {
        const uint16_t *d_wq = resident_weight(wq, static_cast<size_t>(q_dim) * in_dim);
        const uint16_t *d_wk = resident_weight(wk, static_cast<size_t>(kv_dim) * in_dim);
        const uint16_t *d_wv = resident_weight(wv, static_cast<size_t>(kv_dim) * in_dim);
        FusedWs &W = ws();
        FusedWs::grow(W.d_x, W.x_cap, static_cast<size_t>(in_dim));
        FusedWs::grow(W.d_yq, W.yq_cap, static_cast<size_t>(q_dim));
        FusedWs::grow(W.d_yk, W.yk_cap, static_cast<size_t>(kv_dim));
        FusedWs::grow(W.d_yv, W.yv_cap, static_cast<size_t>(kv_dim));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(W.d_x, x, static_cast<size_t>(in_dim) * sizeof(float),
                                       cudaMemcpyHostToDevice));
        const int blocks = q_dim + 2 * kv_dim;
        f16_qkv_coal_kernel<<<blocks, kThreads>>>(d_wq, d_wk, d_wv, W.d_x, W.d_yq, W.d_yk,
                                                  W.d_yv, q_dim, kv_dim, in_dim);
        TINYQWEN_CUDA_CHECK(cudaGetLastError());
        TINYQWEN_CUDA_CHECK(cudaMemcpy(yq, W.d_yq, static_cast<size_t>(q_dim) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(yk, W.d_yk, static_cast<size_t>(kv_dim) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
        TINYQWEN_CUDA_CHECK(cudaMemcpy(yv, W.d_yv, static_cast<size_t>(kv_dim) * sizeof(float),
                                       cudaMemcpyDeviceToHost));
    }

    // 注册进 pair / qkv 两个 f16 注册表，名字与 matvec 主表共享。
    TINYQWEN_MATVEC_F16_PAIR_VARIANT(matvec_f16_pair_fused, "cuda_resident_coal_ws");
    TINYQWEN_MATVEC_QKV_F16_VARIANT(matvec_f16_qkv_fused, "cuda_resident_coal_ws");
} // namespace tinyqwen
