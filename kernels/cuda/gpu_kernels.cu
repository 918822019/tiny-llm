// GPU decode engine 的 kernel 实现（device 指针形态）。接口见 gpu_kernels.cuh。
//
// 设计要点：
//   - 全部接收 device 指针、在给定 stream 上启动，内部不做 H2D/D2H；
//   - matvec 用 warp-per-row（一个 warp 独立归约一整行，8 warp/block 同时
//     算 8 行，纯 shuffle、无 shared mem/__syncthreads）；这一点与
//     matvec_f*_cuda_resident_coal*.cu 的 block-per-row 方案不同——engine
//     自带一份独立实现，改动不影响那批已测好的 dispatch 变体；
//   - 非 matvec 算子逐个对齐 *_ref 的数值（rmsnorm 用 double 累加平方和、
//     rope 用 powf/cosf/sinf、swiglu 用 expf 除法形式），容差见 tests。
// 仅 CUDA 构建编译。

#include "cuda/gpu_kernels.cuh"

#include <cstdio>
#include <cstdlib>

#include <cuda_fp16.h>

namespace tinyqwen {
namespace gpu {
    namespace {
        inline void check(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen gpu: CUDA 调用失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define GPU_CHECK(expr) tinyqwen::gpu::check((expr), #expr)

        constexpr int kBlock = 256; // 通用 block 大小
        constexpr int kWarp = 32;

        // 一个 warp 内 float 求和（shuffle down）。
        __device__ __forceinline__ float warp_sum(float v) {
            for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(~0u, v, off);
            return v;
        }
        // block 内 double 求和（rmsnorm 平方和 / attention 点积用 double，对齐 ref）。
        __device__ __forceinline__ double warp_sum_d(double v) {
            for (int off = kWarp / 2; off > 0; off >>= 1) v += __shfl_down_sync(~0u, v, off);
            return v;
        }
        __device__ double block_sum_d(double v, double *shared /* >= blockDim.x/32 */) {
            const int lane = threadIdx.x & (kWarp - 1);
            const int wid = threadIdx.x / kWarp;
            v = warp_sum_d(v);
            if (lane == 0) shared[wid] = v;
            __syncthreads();
            const int nw = blockDim.x / kWarp;
            if (wid == 0) {
                double t = (lane < nw) ? shared[lane] : 0.0;
                t = warp_sum_d(t);
                if (lane == 0) shared[0] = t;
            }
            __syncthreads();
            return shared[0];
        }

        constexpr int kWarpsPerBlock = kBlock / kWarp; // 8

        // coalesced 行点积：warp 内 32 线程沿 in_dim 跨步（stride 32）乘加，
        // 一个 warp 独立算完一整行，只用 warp-shuffle，不需要 shared mem/
        // __syncthreads（一个 block 里 8 个 warp 各管各的行，互不依赖）。
        template <typename W>
        __device__ __forceinline__ float row_partial_warp(const W *w_row, const float *x,
                                                          int in_dim, int lane) {
            float acc = 0.0f;
            for (int i = lane; i < in_dim; i += kWarp) {
                acc += static_cast<float>(w_row[i]) * x[i];
            }
            return acc;
        }
        // f16 特化：__half2float。
        template <>
        __device__ __forceinline__ float
        row_partial_warp<__half>(const __half *w_row, const float *x, int in_dim, int lane) {
            float acc = 0.0f;
            for (int i = lane; i < in_dim; i += kWarp) {
                acc += __half2float(w_row[i]) * x[i];
            }
            return acc;
        }
    } // namespace

    // ========================================================================
    // matvec：warp-per-row（一个 warp 独立算一整行），一个 block 装
    // kWarpsPerBlock=8 个 warp、同时算 8 行。之前是 block-per-row（256 线程
    // 归约 896 个元素，每线程只摸 ~3.5 个、还要两级 shared mem 归约+两次
    // __syncthreads）；k/v_proj 这类小矩阵（128/256 行）在那个方案下线程
    // 利用率低。warp-per-row 把归约收窄到 32 线程/warp-shuffle-only，每线程
    // 摸 in_dim/32（896 时 28 个）个元素，且不同 warp 之间零同步开销，单次
    // launch 处理的行数不变、只是每个 block 干更多活——launch 次数不变，
    // 但 block 数从"总行数"降到"总行数/8"，配合更高的每线程算力利用率。
    // single / pair / qkv 共享同一套 warp 内归约。
    // ========================================================================
    template <typename W>
    __global__ void matvec_single_kernel(const W *w, const float *x, float *y, int out_dim,
                                         int in_dim) {
        const int lane = threadIdx.x & (kWarp - 1);
        const int row = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp;
        if (row >= out_dim) return;
        const W *w_row = w + static_cast<size_t>(row) * in_dim;
        const float sum = warp_sum(row_partial_warp(w_row, x, in_dim, lane));
        if (lane == 0) y[row] = sum;
    }

    template <typename W>
    __global__ void matvec_pair_kernel(const W *w1, const W *w2, const float *x, float *y1,
                                       float *y2, int out_dim, int in_dim) {
        const int lane = threadIdx.x & (kWarp - 1);
        const int grow = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp; // [0, 2*out_dim)
        if (grow >= 2 * out_dim) return;
        const W *W_sel;
        float *out;
        int row;
        if (grow < out_dim) { W_sel = w1; out = y1; row = grow; }
        else { W_sel = w2; out = y2; row = grow - out_dim; }
        const W *w_row = W_sel + static_cast<size_t>(row) * in_dim;
        const float sum = warp_sum(row_partial_warp(w_row, x, in_dim, lane));
        if (lane == 0) out[row] = sum;
    }

    template <typename W>
    __global__ void matvec_qkv_kernel(const W *wq, const W *wk, const W *wv, const float *x,
                                      float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                                      int in_dim) {
        const int lane = threadIdx.x & (kWarp - 1);
        const int grow = blockIdx.x * kWarpsPerBlock + threadIdx.x / kWarp; // [0, q_dim+2*kv_dim)
        const int total = q_dim + 2 * kv_dim;
        if (grow >= total) return;
        const W *W_sel;
        float *out;
        int row;
        if (grow < q_dim) { W_sel = wq; out = yq; row = grow; }
        else if (grow < q_dim + kv_dim) { W_sel = wk; out = yk; row = grow - q_dim; }
        else { W_sel = wv; out = yv; row = grow - q_dim - kv_dim; }
        const W *w_row = W_sel + static_cast<size_t>(row) * in_dim;
        const float sum = warp_sum(row_partial_warp(w_row, x, in_dim, lane));
        if (lane == 0) out[row] = sum;
    }

    // ---- matvec launchers ----
    // 启动后立即查启动错误（配置非法等会在这里 fail loud；运行期错误由
    // engine 末尾的 cudaDeviceSynchronize 兜底）。变参宏：kernel 调用的实参
    // 里有逗号，单参宏会被预处理器拆开。grid = ceil(总行数 / kWarpsPerBlock)，
    // block 仍是 kBlock=256（8 个 warp）。
#define GPU_LAUNCH(...)                                                              \
    do {                                                                             \
        __VA_ARGS__;                                                                 \
        GPU_CHECK(cudaGetLastError());                                               \
    } while (0)
#define GPU_ROW_BLOCKS(total_rows) (((total_rows) + kWarpsPerBlock - 1) / kWarpsPerBlock)

    void matvec_f16(cudaStream_t s, const uint16_t *w, const float *x, float *y, int out_dim,
                    int in_dim) {
        GPU_LAUNCH(matvec_single_kernel<__half><<<GPU_ROW_BLOCKS(out_dim), kBlock, 0, s>>>(
                reinterpret_cast<const __half *>(w), x, y, out_dim, in_dim));
    }
    void matvec_f16_pair(cudaStream_t s, const uint16_t *w1, const uint16_t *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        GPU_LAUNCH(matvec_pair_kernel<__half><<<GPU_ROW_BLOCKS(2 * out_dim), kBlock, 0, s>>>(
                reinterpret_cast<const __half *>(w1), reinterpret_cast<const __half *>(w2), x, y1,
                y2, out_dim, in_dim));
    }
    void matvec_f16_qkv(cudaStream_t s, const uint16_t *wq, const uint16_t *wk,
                        const uint16_t *wv, const float *x, float *yq, float *yk, float *yv,
                        int q_dim, int kv_dim, int in_dim) {
        GPU_LAUNCH(matvec_qkv_kernel<__half>
                           <<<GPU_ROW_BLOCKS(q_dim + 2 * kv_dim), kBlock, 0, s>>>(
                                   reinterpret_cast<const __half *>(wq),
                                   reinterpret_cast<const __half *>(wk),
                                   reinterpret_cast<const __half *>(wv), x, yq, yk, yv, q_dim,
                                   kv_dim, in_dim));
    }
    void matvec_f32(cudaStream_t s, const float *w, const float *x, float *y, int out_dim,
                    int in_dim) {
        GPU_LAUNCH(matvec_single_kernel<float><<<GPU_ROW_BLOCKS(out_dim), kBlock, 0, s>>>(
                w, x, y, out_dim, in_dim));
    }
    void matvec_f32_pair(cudaStream_t s, const float *w1, const float *w2, const float *x,
                         float *y1, float *y2, int out_dim, int in_dim) {
        GPU_LAUNCH(matvec_pair_kernel<float><<<GPU_ROW_BLOCKS(2 * out_dim), kBlock, 0, s>>>(
                w1, w2, x, y1, y2, out_dim, in_dim));
    }
    void matvec_f32_qkv(cudaStream_t s, const float *wq, const float *wk, const float *wv,
                        const float *x, float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                        int in_dim) {
        GPU_LAUNCH(matvec_qkv_kernel<float><<<GPU_ROW_BLOCKS(q_dim + 2 * kv_dim), kBlock, 0, s>>>(
                wq, wk, wv, x, yq, yk, yv, q_dim, kv_dim, in_dim));
    }

    // ========================================================================
    // embed_lookup：hidden = embed 表第 token 行（f16 时逐元素转 fp32）。
    // ========================================================================
    __global__ void embed_f32_kernel(float *hidden, const float *embed, const int *token_ptr,
                                     int hidden_dim) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int token = *token_ptr;
        if (i < hidden_dim) hidden[i] = embed[static_cast<size_t>(token) * hidden_dim + i];
    }
    __global__ void embed_f16_kernel(float *hidden, const __half *embed, const int *token_ptr,
                                     int hidden_dim) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const int token = *token_ptr;
        if (i < hidden_dim)
            hidden[i] = __half2float(embed[static_cast<size_t>(token) * hidden_dim + i]);
    }
    void embed_lookup(cudaStream_t s, float *hidden, const void *embed, const int *token,
                      int hidden_dim, bool is_f16) {
        const int blocks = (hidden_dim + kBlock - 1) / kBlock;
        if (is_f16)
            GPU_LAUNCH(embed_f16_kernel<<<blocks, kBlock, 0, s>>>(
                    hidden, reinterpret_cast<const __half *>(embed), token, hidden_dim));
        else
            GPU_LAUNCH(embed_f32_kernel<<<blocks, kBlock, 0, s>>>(
                    hidden, reinterpret_cast<const float *>(embed), token, hidden_dim));
    }

    // ========================================================================
    // rmsnorm：double 累加平方和（对齐 ref）。一个 block（n<=896 够用）。
    // ========================================================================
    __global__ void rmsnorm_kernel(float *y, const float *x, const float *w, int n, float eps) {
        __shared__ double ssum[kBlock / kWarp];
        __shared__ float sscale;
        double acc = 0.0;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const double xi = static_cast<double>(x[i]);
            acc += xi * xi;
        }
        const double total = block_sum_d(acc, ssum);
        if (threadIdx.x == 0) {
            const float mean_sq = static_cast<float>(total / n);
            sscale = 1.0f / std::sqrt(mean_sq + eps);
        }
        __syncthreads();
        const float scale = sscale;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            y[i] = (x[i] * scale) * w[i];
        }
    }
    void rmsnorm(cudaStream_t s, float *y, const float *x, const float *w, int n, float eps) {
        GPU_LAUNCH(rmsnorm_kernel<<<1, kBlock, 0, s>>>(y, x, w, n, eps));
    }

    // ========================================================================
    // bias_add / residual_add：逐元素。
    // ========================================================================
    __global__ void bias_add_kernel(float *x, const float *bias, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] += bias[i];
    }
    void bias_add(cudaStream_t s, float *x, const float *bias, int n) {
        GPU_LAUNCH(bias_add_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(x, bias, n));
    }
    __global__ void residual_add_kernel(float *hidden, const float *delta, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) hidden[i] += delta[i];
    }
    void residual_add(cudaStream_t s, float *hidden, const float *delta, int n) {
        GPU_LAUNCH(residual_add_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(hidden, delta,
                                                                                    n));
    }

    // ========================================================================
    // rope：rotate-half。每个线程处理一个 (head, 配对 i)；cos/sin 用
    // powf/cosf/sinf（与 rope_ref 的单精度一致）。q、k 各启动一段。
    // ========================================================================
    __global__ void rope_kernel(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                                const int *pos_ptr, float theta) {
        const int half = head_dim / 2;
        const int total = (n_heads + n_kv_heads) * half; // 所有 (head, i) 配对
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= total) return;
        const int pos = *pos_ptr;
        const int i = t % half;    // 配对内下标
        const int h_all = t / half; // 第几个 (head)（先 q 后 k）
        float *base;
        if (h_all < n_heads) base = q + static_cast<size_t>(h_all) * head_dim;
        else base = k + static_cast<size_t>(h_all - n_heads) * head_dim;

        const float exponent = -static_cast<float>(2 * i) / static_cast<float>(head_dim);
        const float inv_freq = powf(theta, exponent);
        const float angle = static_cast<float>(pos) * inv_freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = base[i];
        const float x1 = base[i + half];
        base[i] = x0 * c - x1 * s;
        base[i + half] = x1 * c + x0 * s;
    }
    void rope(cudaStream_t s, float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
              const int *pos, float theta) {
        const int half = head_dim / 2;
        const int total = (n_heads + n_kv_heads) * half;
        GPU_LAUNCH(rope_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(
                q, k, n_heads, n_kv_heads, head_dim, pos, theta));
    }

    // ========================================================================
    // kv_append：把 k_in/v_in 追加进某层 plane 的 pos 槽位（索引对齐 qwen_model）。
    // 共 n_kv_heads*head_dim 个 float/每个 k 和 v。
    // ========================================================================
    __global__ void kv_append_kernel(float *k_plane, float *v_plane, const float *k_in,
                                     const float *v_in, const int *pos_ptr, int n_kv_heads,
                                     int max_seq_len, int head_dim) {
        const int per = n_kv_heads * head_dim; // k（或 v）的元素总数
        const int t = blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= 2 * per) return;
        const int pos = *pos_ptr;
        const bool is_v = t >= per;
        const int idx = is_v ? t - per : t; // [0, per)
        const int h = idx / head_dim;
        const int d = idx % head_dim;
        const size_t head_plane = static_cast<size_t>(max_seq_len) * head_dim;
        const size_t dst = static_cast<size_t>(h) * head_plane +
                           static_cast<size_t>(pos) * head_dim + d;
        if (is_v) v_plane[dst] = v_in[idx];
        else k_plane[dst] = k_in[idx];
    }
    void kv_append(cudaStream_t s, float *k_plane, float *v_plane, const float *k_in,
                   const float *v_in, const int *pos, int n_kv_heads, int max_seq_len,
                   int head_dim) {
        const int per = n_kv_heads * head_dim;
        const int total = 2 * per;
        GPU_LAUNCH(kv_append_kernel<<<(total + kBlock - 1) / kBlock, kBlock, 0, s>>>(
                k_plane, v_plane, k_in, v_in, pos, n_kv_heads, max_seq_len, head_dim));
    }

    // ========================================================================
    // attention_decode：一个 block 一个 query head，online softmax 沿 seq_len
    // 串行（与 ref 同序），head_dim 维并行。对齐 attention_decode_ref。
    // ========================================================================
    __global__ void attention_kernel(float *out, const float *q, const float *k_cache,
                                     const float *v_cache, const int *pos_ptr, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim, float scale) {
        const int h = blockIdx.x;
        if (h >= n_heads) return;
        const int seq_len = *pos_ptr + 1;
        const int i = threadIdx.x; // 负责 head_dim 里的第 i 维（要求 blockDim.x>=head_dim）
        const int heads_per_kv = n_heads / n_kv_heads;
        const int kv = h / heads_per_kv;
        const size_t kv_stride = static_cast<size_t>(max_seq_len) * head_dim;
        const float *qh = q + static_cast<size_t>(h) * head_dim;
        const float *kh = k_cache + static_cast<size_t>(kv) * kv_stride;
        const float *vh = v_cache + static_cast<size_t>(kv) * kv_stride;
        float *oh = out + static_cast<size_t>(h) * head_dim;

        __shared__ float sdot;              // 广播每个 t 的点积
        __shared__ double smd[kBlock / kWarp]; // 点积 double 归约用（8 个 double）

        const bool active = i < head_dim;
        const float qi = active ? qh[i] : 0.0f;
        float oh_i = 0.0f;
        float m = -INFINITY; // 与 ref 一致（-inf）；expf(-inf)=0，首轮安全
        float l = 0.0f;

        for (int t = 0; t < seq_len; ++t) {
            const float *kt = kh + static_cast<size_t>(t) * head_dim;
            const float *vt = vh + static_cast<size_t>(t) * head_dim;
            // 点积（double 累加，对齐 ref），block 归约后广播。
            const float prod = active ? qi * kt[i] : 0.0f;
            const double dsum = block_sum_d(static_cast<double>(prod), smd);
            if (threadIdx.x == 0) sdot = static_cast<float>(dsum) * scale;
            __syncthreads();
            const float s = sdot;
            // online softmax 更新（所有线程同一份标量）。
            const float m_new = s > m ? s : m;
            const float rescale = expf(m - m_new);
            const float p = expf(s - m_new);
            if (active) oh_i = oh_i * rescale + p * vt[i];
            l = l * rescale + p;
            m = m_new;
            __syncthreads(); // 确保所有线程读完 sdot 再进下一轮覆写
        }
        const float inv_l = 1.0f / l;
        if (active) oh[i] = oh_i * inv_l;
    }
    void attention_decode(cudaStream_t s, float *out, const float *q, const float *k_cache,
                          const float *v_cache, const int *pos, int max_seq_len, int n_heads,
                          int n_kv_heads, int head_dim, float scale) {
        // block 大小取 >= head_dim 的最小 warp（32）倍数——之前固定用 kBlock=256，
        // 但 attention_kernel 每个 timestep 都要做一次 block 级归约（block_sum_d，
        // 两次 __syncthreads），head_dim=64 时 256 线程里 192 个白白陪跑、归约树
        // 还多一层，seq_len 次迭代的开销就被放大了；profile 显示这是 engine 里
        // 单项占比最大的 kernel（~35%/token）。收紧到刚好覆盖 head_dim 的 warp
        // 数（64→64），归约收窄到 2 个 warp 合并，无浪费线程。
        const int block_size = ((head_dim + kWarp - 1) / kWarp) * kWarp;
        GPU_LAUNCH(attention_kernel<<<n_heads, block_size, 0, s>>>(
                out, q, k_cache, v_cache, pos, max_seq_len, n_heads, n_kv_heads, head_dim,
                scale));
    }

    // ========================================================================
    // swiglu：gate = silu(gate)*up，除法形式对齐 swiglu_ref。
    // ========================================================================
    __global__ void swiglu_kernel(float *gate, const float *up, int n) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const float denom = 1.0f + expf(-gate[i]);
        const float sil = gate[i] / denom;
        gate[i] = sil * up[i];
    }
    void swiglu(cudaStream_t s, float *gate, const float *up, int n) {
        GPU_LAUNCH(swiglu_kernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, s>>>(gate, up, n));
    }

    // ========================================================================
    // argmax：单 block 跨步扫描 + (value,index) 归约，严格 > 取首个（对齐 ref）。
    // ========================================================================
    struct ValIdx { float v; int idx; };
    __device__ __forceinline__ ValIdx better(ValIdx a, ValIdx b) {
        // 值大者赢；值相等取下标小者（= 先出现者），与 argmax_ref 的严格 > 一致。
        if (a.v > b.v) return a;
        if (b.v > a.v) return b;
        return (a.idx <= b.idx) ? a : b;
    }
    __global__ void argmax_kernel(const float *logits, int *out_idx, int n) {
        __shared__ ValIdx sbest[kBlock / kWarp];
        ValIdx local{-1e30f, 0};
        // 每个线程负责 i = tid, tid+blockDim, ...（升序，配合严格 > 保留首个）。
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const float v = logits[i];
            if (v > local.v) local = ValIdx{v, i};
        }
        // warp 内归约。
        const int lane = threadIdx.x & (kWarp - 1);
        const int wid = threadIdx.x / kWarp;
        for (int off = kWarp / 2; off > 0; off >>= 1) {
            ValIdx other;
            other.v = __shfl_down_sync(~0u, local.v, off);
            other.idx = __shfl_down_sync(~0u, local.idx, off);
            local = better(local, other);
        }
        if (lane == 0) sbest[wid] = local;
        __syncthreads();
        const int nw = blockDim.x / kWarp;
        if (wid == 0) {
            ValIdx t = (lane < nw) ? sbest[lane] : ValIdx{-1e30f, INT32_MAX};
            for (int off = nw / 2; off > 0; off >>= 1) {
                ValIdx other;
                other.v = __shfl_down_sync(~0u, t.v, off);
                other.idx = __shfl_down_sync(~0u, t.idx, off);
                t = better(t, other);
            }
            if (lane == 0) *out_idx = t.idx;
        }
    }
    void argmax(cudaStream_t s, const float *logits, int *out_idx, int n) {
        GPU_LAUNCH(argmax_kernel<<<1, kBlock, 0, s>>>(logits, out_idx, n));
    }

} // namespace gpu
} // namespace tinyqwen
