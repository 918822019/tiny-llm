// =============================================================================
// benchmarks/bench_metal_gemm.mm —— MPS vs 自写 tiled GEMM 微基准
// =============================================================================
//
// 目的：把 prefill 的线性投影 GEMM 从 MPS MPSMatrixMultiplication 换成自写
// Metal compute kernel，看能不能突破 MPS 只有峰值 57% 的现状。
//
// 为什么要独立微基准而不是直接改 runtime/metal_prefill.mm：
//   - 隔离：kernel 可以独立迭代，不用动 1300 行的引擎
//   - 同场 A/B：同一份数据跑 MPS 与自写 kernel，误差与耗时直接可比
//   - 零风险：工作正常的引擎不受影响
// 验证通过后再接进 metal_prefill.mm。
//
// 测的是 **Y[n,out] = X[n,in] @ W[out,in]^T** —— Transformer 线性投影的形状，
// 与 HF 权重布局（[out,in] 行主序）一致，所以不需要预先转置权重。
//
// 测量纪律（见 AGENTS.md 坑 #7）：
//   - 计时用 GPU 侧 GPUStartTime/GPUEndTime，不含 CPU 编排 —— 这样即使
//     agent 自己把 CPU 占满也不污染数字（端到端墙钟会被污染）；
//   - 取 min 而不是中位数：干扰是加性的，min 更稳；
//   - 输出离散度（max/min），>1.5 的行不可用于归因。
//
// 数值口径：自写 kernel vs MPS 逐元素 max_abs_err（MPS 是当前已验证路径），
// 另外附一份小规模 CPU fp32 参考做绝对正确性检查。
// =============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

    // =========================================================================
    // tile 常量（取自 ggml src/ggml-metal/kernels/mul_mm.metal 的经典实现）
    //
    //   NR0 = 输出 M 方向（out）每线程组的行数
    //   NR1 = 输出 N 方向（n）每线程组的列数
    //   NK  = K 方向每步深度
    // （NK/16=2 与 NK/8=4 这两个派生量直接内联在 shader 源码里，host 侧用不到。）
    //
    // 一个线程组 = 4 simdgroup × 32 lane = 128 线程，算 NR0×NR1 = 2048 个输出，
    // 即每线程 16 个累加器。sgitg%2 选 32 行的 out 分组，sgitg>>1 选 16 列的 n 分组，
    // 所以每个 simdgroup 拥有 32×16 的输出子块 = 8 个 8x8 tile。
    // =========================================================================
    constexpr int kNR0 = 64;
    constexpr int kNR1 = 32;
    constexpr int kNK = 32;
    constexpr int kThreads = 128;  // 4 simdgroup × 32 lane

    // 片上暂存容量（float 计数）。两个变体共用同一块分配，f16 变体只用一半。
    //   sa: NR0 × NK = 2048（**转置**存放：[kblock][rowblock][k_within][row_within]）
    //   sb: NR1 × NK = 1024（正常朝向：[kblock][rowblock][row_within][k_within]）
    constexpr int kSaFloats = kNR0 * kNK;
    constexpr int kSbFloats = kNR1 * kNK;
    constexpr size_t kShmemBytes = (kSaFloats + kSbFloats) * sizeof(float); // 12 KB

    // =========================================================================
    // MSL 源码
    //
    // 两个变体，差别只在片上暂存的 dtype：
    //   gemm_wt_tiled_f32 —— 权重与激活都以 fp32 暂存，simdgroup_float8x8 乘加。
    //                        精度与当前 MPS 路径一致（权重 fp16→fp32 是精确的，
    //                        fp16 尾数 11 位 < fp32 的 24 位）。代价是片上 12KB。
    //   gemm_wt_tiled_f16 —— 都以 fp16 暂存，half 乘 half、fp32 累加。
    //                        片上只用 6KB（占用率翻倍），但激活 fp32→fp16 丢
    //                        13 位尾数，相对误差 ~5e-4。
    // 谁更快由实测决定，不是猜。
    // =========================================================================
    constexpr const char *kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// 片上暂存的 8x8 block swizzle
//
// simdgroup_load(d, ptr, 8) 按行跨度 8 读一个 8x8 块，所以片上必须把 tile
// 切成 8x8 块、每块 64 个连续元素存放。索引公式：
//
//   sa（权重，转置）：sa[64*ib + 8*ly + lx]
//       ib = 8*sx + sy   sx = k 方向第几个 8 元素块 (0..3)
//                        sy = out 方向第几个 8 行块 (0..7)
//       ly = k 块内偏移   lx = out 行块内偏移
//       → 块内朝向是 [k][out]，即 A^T，因为要算的是 X @ W^T
//
//   sb（激活，正常）：sb[64*ib + 8*ly + lx]
//       ib = 4*sx + sy   sx = k 块 (0..3)   sy = n 方向第几个 8 行块 (0..3)
//       ly = n 行块内偏移  lx = k 块内偏移
//       → 块内朝向是 [n][k]
//
// 于是 simdgroup_multiply_accumulate(mc, mb, ma, mc) = [n][k] × [k][out] = [n][out]，
// 正好是我们要的输出朝向。
// ---------------------------------------------------------------------------

// fp32 暂存变体
kernel void gemm_wt_tiled_f32(device const half  *W,
                              device const float *X,
                              device       float *Y,
                              constant int &out_cnt, constant int &in_cnt, constant int &n_cnt,
                              constant int &w_stride, constant int &x_stride, constant int &y_stride,
                              threadgroup char *shmem [[threadgroup(0)]],
                              uint3  tgpig [[threadgroup_position_in_grid]],
                              ushort tiitg [[thread_index_in_threadgroup]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float *sa = (threadgroup float *)(shmem);
    threadgroup float *sb = (threadgroup float *)(shmem + 4 * 2048);

    const int r0 = tgpig.y * 64; // out 方向起点
    const int r1 = tgpig.x * 32; // n 方向起点

    // 本线程负责的 out 行与 k 块
    const short lr0 = (short)(tiitg / 2);
    const short il0 = (short)(tiitg % 2);
    // 本线程负责的 n 行
    const short lr1 = (short)(tiitg / 4);
    const short iy  = (short)(8 * (tiitg % 4));

    device const half  *wx = W + (size_t)(r0 + lr0) * w_stride;
    device const float *xr = X + (size_t)(r1 + lr1) * x_stride + iy;

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (int loop_k = 0; loop_k < in_cnt; loop_k += 32) {
        // ---- PHASE 1: 暂存权重 tile（fp16 读入 → fp32 写片上，转换精确）----
        // 一次读 16 个连续 half：4 个 half4 向量化加载，而不是 16 次 2 字节标量
        // 加载（标量版是这个 kernel 最初的主要低效来源）。写片上仍必须逐元素 ——
        // 目标是 8x8 block swizzle，落点是散的。
        // 对齐账：字节偏移 = ((r0+lr0)*w_stride + loop_k + 16*il0)*2，
        // w_stride 为偶、loop_k 是 32 的倍数、16*il0 是 16 的倍数 → 32 字节对齐，
        // 满足 half4 的 8 字节要求。in_cnt 是 32 的倍数时不会越界。
        half wv[16];
        {
            device const half *wp = wx + loop_k + 16 * il0;
            const half4 v0 = *(device const half4 *)(wp + 0);
            const half4 v1 = *(device const half4 *)(wp + 4);
            const half4 v2 = *(device const half4 *)(wp + 8);
            const half4 v3 = *(device const half4 *)(wp + 12);
            wv[0] = v0.x; wv[1] = v0.y; wv[2] = v0.z; wv[3] = v0.w;
            wv[4] = v1.x; wv[5] = v1.y; wv[6] = v1.z; wv[7] = v1.w;
            wv[8] = v2.x; wv[9] = v2.y; wv[10] = v2.z; wv[11] = v2.w;
            wv[12] = v3.x; wv[13] = v3.y; wv[14] = v3.z; wv[15] = v3.w;
        }

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short)((tiitg / 2) / 8);
            const short lx = (short)((tiitg / 2) % 8);
            const short ly = i % 8;
            const short ib = 8 * sx + sy;

            const int k = loop_k + 16 * il0 + i;
            const float v = (k < in_cnt) ? (float)wv[i] : 0.0f;
            *(sa + 64 * ib + 8 * ly + lx) = v;
        }

        // ---- PHASE 1b: 暂存激活 tile（fp32，两个 float4 搬 8 个元素）----
        {
            const short sx = (short)(tiitg % 4);
            const short sy = (short)((tiitg / 4) / 8);
            const short ly = (short)((tiitg / 4) % 8);
            const short ib = 4 * sx + sy;

            const int k = loop_k + iy;
            threadgroup float *dstp = sb + 64 * ib + 8 * ly;
            // xr 里已经 baked 了 iy，但 loop_k 必须在这里加 —— 否则每个 K 迭代
            // 都重读同一批 8 个元素（这正是最初那版的大误差来源）。
            if (k + 8 <= in_cnt) {
                *(threadgroup float4 *)(dstp + 0) = *(device const float4 *)(xr + loop_k + 0);
                *(threadgroup float4 *)(dstp + 4) = *(device const float4 *)(xr + loop_k + 4);
            } else {
                for (short i = 0; i < 8; ++i)
                    dstp[i] = (k + i < in_cnt) ? xr[loop_k + i] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- PHASE 2: simdgroup 外积累加 ----
        threadgroup const float *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float *lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i)
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- PHASE 3: 写回。mc[i] 是 [n][out] 的 8x8 块，行跨度 = y_stride ----
    // 基址：out 方向 r0 + 32*(sgitg&1)，n 方向 r1 + 16*(sgitg>>1)
    device float *C = Y + (r0 + 32 * (sgitg & 1)) + (size_t)(r1 + 16 * (sgitg >> 1)) * y_stride;

    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], C + 8 * (i % 4) + (size_t)8 * y_stride * (i / 4), y_stride, 0, false);
    }
}

// fp16 暂存变体：片上 6KB，half × half → fp32 累加
kernel void gemm_wt_tiled_f16(device const half  *W,
                              device const float *X,
                              device       float *Y,
                              constant int &out_cnt, constant int &in_cnt, constant int &n_cnt,
                              constant int &w_stride, constant int &x_stride, constant int &y_stride,
                              threadgroup char *shmem [[threadgroup(0)]],
                              uint3  tgpig [[threadgroup_position_in_grid]],
                              ushort tiitg [[thread_index_in_threadgroup]],
                              ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + 2 * 2048);

    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;

    const short lr0 = (short)(tiitg / 2);
    const short il0 = (short)(tiitg % 2);
    const short lr1 = (short)(tiitg / 4);
    const short iy  = (short)(8 * (tiitg % 4));

    device const half  *wx = W + (size_t)(r0 + lr0) * w_stride;
    device const float *xr = X + (size_t)(r1 + lr1) * x_stride + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (int loop_k = 0; loop_k < in_cnt; loop_k += 32) {
        // 同 f32 变体：4 个 half4 向量化读入。这里不需要 dtype 转换（片上也是 half）。
        half wv[16];
        {
            device const half *wp = wx + loop_k + 16 * il0;
            const half4 v0 = *(device const half4 *)(wp + 0);
            const half4 v1 = *(device const half4 *)(wp + 4);
            const half4 v2 = *(device const half4 *)(wp + 8);
            const half4 v3 = *(device const half4 *)(wp + 12);
            wv[0] = v0.x; wv[1] = v0.y; wv[2] = v0.z; wv[3] = v0.w;
            wv[4] = v1.x; wv[5] = v1.y; wv[6] = v1.z; wv[7] = v1.w;
            wv[8] = v2.x; wv[9] = v2.y; wv[10] = v2.z; wv[11] = v2.w;
            wv[12] = v3.x; wv[13] = v3.y; wv[14] = v3.z; wv[15] = v3.w;
        }

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short)((tiitg / 2) / 8);
            const short lx = (short)((tiitg / 2) % 8);
            const short ly = i % 8;
            const short ib = 8 * sx + sy;

            const int k = loop_k + 16 * il0 + i;
            const half v = (k < in_cnt) ? wv[i] : (half)0.0f;
            *(sa + 64 * ib + 8 * ly + lx) = v;
        }

        {
            const short sx = (short)(tiitg % 4);
            const short sy = (short)((tiitg / 4) / 8);
            const short ly = (short)((tiitg / 4) % 8);
            const short ib = 4 * sx + sy;

            const int k = loop_k + iy;
            threadgroup half *dstp = sb + 64 * ib + 8 * ly;
            // 同上：loop_k 必须加在 xr 的索引上
            for (short i = 0; i < 8; ++i)
                dstp[i] = (k + i < in_cnt) ? (half)xr[loop_k + i] : (half)0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i)
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *C = Y + (r0 + 32 * (sgitg & 1)) + (size_t)(r1 + 16 * (sgitg >> 1)) * y_stride;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], C + 8 * (i % 4) + (size_t)8 * y_stride * (i / 4), y_stride, 0, false);
    }
}

// ---------------------------------------------------------------------------
// GEMV：Y[1, out] = X[1, in] @ W[out, in]^T
//
// 这是 lm_head 在 prefill 里真正需要的形状 —— 只有末位 token 的 logits 会被
// 消费，其余 n-1 行纯浪费。seq=512 时全行 GEMM 要 65 ms，而 N=1 的版本是
// **带宽受限**而不是算力受限：
//   权重 151936×1024 fp16 = 311 MB → 120 GB/s 下 ~2.6 ms
//   算力 2×151936×1024 = 3.1e8 FLOP → 4.6 TFLOPS 下 0.07 ms
// 所以只要把 311 MB 读干净就到底了。
//
// 设计要点：
//   1. **一个线程一个输出行**：不需要跨线程归约，也没有 barrier（除了载入 x）。
//      行数 = out（151936），线程数充裕，负载天然均衡。
//   2. **权重按 uint4（16 字节 = 8 个 half）向量化读**：w_stride 是偶数、
//      k 是 8 的倍数 → 地址 16 字节对齐，满足 uint4 要求。
//   3. **x 协作载入 threadgroup memory**：x 只有 in 个 float（4 KB），但每个
//      线程都要读全部 in 个。放片上后同一线程组内所有线程在同一 k 上读同一个
//      xs[k+i] —— 这是**广播读**，硬件上几乎免费；比每线程从 device 重读
//      1024 次省得多。
//   4. half→float 用 as_type<half2> 做位重解释（uint 是 4 字节 = 2 个 half），
//      不做算术转换指令。
// ---------------------------------------------------------------------------
kernel void gemv_wt(device const half  *W,
                    device const float *x,
                    device       float *y,
                    constant int &out_cnt, constant int &in_cnt, constant int &w_stride,
                    threadgroup float *xs [[threadgroup(0)]],
                    uint tid   [[thread_position_in_grid]],
                    uint tiitg [[thread_index_in_threadgroup]],
                    uint nthr  [[threads_per_threadgroup]]) {
    // 协作载入 x 到片上。必须所有线程都参与（包括 tid >= out_cnt 的），
    // 否则后面的 barrier 会挂。
    for (uint i = tiitg; i < (uint)in_cnt; i += nthr) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= (uint)out_cnt) return;

    device const half *wr = W + (size_t)tid * w_stride;
    float acc = 0.0f;

    for (int k = 0; k + 8 <= in_cnt; k += 8) {
        const uint4 p = *(device const uint4 *)(wr + k);
        const half2 h0 = as_type<half2>(p.x);
        const half2 h1 = as_type<half2>(p.y);
        const half2 h2 = as_type<half2>(p.z);
        const half2 h3 = as_type<half2>(p.w);
        acc += (float)h0.x * xs[k + 0] + (float)h0.y * xs[k + 1];
        acc += (float)h1.x * xs[k + 2] + (float)h1.y * xs[k + 3];
        acc += (float)h2.x * xs[k + 4] + (float)h2.y * xs[k + 5];
        acc += (float)h3.x * xs[k + 6] + (float)h3.y * xs[k + 7];
    }
    // in_cnt 不是 8 的倍数时的尾巴
    for (int k = (in_cnt / 8) * 8; k < in_cnt; ++k) acc += (float)wr[k] * xs[k];

    y[tid] = acc;
}

// ---------------------------------------------------------------------------
// INT4 tiled GEMM：Y[n,out] = X[n,in] @ dequant(W[out,in])^T
//
// MPS 没有 int4 GEMM 路径，所以这里**必须**自写 —— 与 fp16 那条"可选但打不过
// MPS"的情况性质完全不同。
//
// 权重格式（docs/weight_format.md 的 INT4 packing，dtype=3）：
//   非对称 uint4 [0,15]，per-group scale+zero，group_size 典型 64。
//   每组 36 字节：[scale_fp16(2B) | zero_fp16(2B) | packed_uint4(32B)]，
//   低 nibble 在前（偶数下标元素占低 4 bit）。
//   反量化：(q - zero) × scale。
//
// 关键对齐账（决定用哪种向量读）：
//   行跨度 = (in/group_size)*36 字节；in=1024,gs=64 → 576，是 8 的倍数。
//   某行某组的 packed 起点 = 576*r + 36*g + 4 + p/2。
//   36*g + 4 在 g 偶数时 ≡ 4 (mod 8)、g 奇数时 ≡ 0 (mod 8) —— **不稳定 8 字节
//   对齐，但恒定 4 字节对齐**（p/2 是 8 的倍数，不改变 mod 4）。
//   所以只能读 2 个 uint（4 字节对齐），不能读 uint2（要 8 字节对齐）——
//   用 uint2 会在偶数组上崩或读到错数据。
//
// 反量化开销可忽略：每个线程每 K 步反量化 16 个元素，供 64×32×32/128 = 512 个
// FMA 使用，即 ~0.03 次反量化/FMA。
// ---------------------------------------------------------------------------
kernel void gemm_wt_tiled_i4(device const uchar *W,
                             device const float *X,
                             device       float *Y,
                             constant int &out_cnt, constant int &in_cnt, constant int &n_cnt,
                             constant int &grp, constant int &w_packed_stride,
                             constant int &x_stride, constant int &y_stride,
                             threadgroup char *shmem [[threadgroup(0)]],
                             uint3  tgpig [[threadgroup_position_in_grid]],
                             ushort tiitg [[thread_index_in_threadgroup]],
                             ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float *sa = (threadgroup float *)(shmem);
    threadgroup float *sb = (threadgroup float *)(shmem + 4 * 2048);

    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 32;

    const short lr0 = (short)(tiitg / 2);
    const short il0 = (short)(tiitg % 2);
    const short lr1 = (short)(tiitg / 4);
    const short iy  = (short)(8 * (tiitg % 4));

    device const uchar *wrow = W + (size_t)(r0 + lr0) * w_packed_stride;
    device const float *xr = X + (size_t)(r1 + lr1) * x_stride + iy;

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (int loop_k = 0; loop_k < in_cnt; loop_k += 32) {
        // ---- PHASE 1: 读 int4 → in-register 反量化 → fp32 写片上 ----
        const int k_abs = loop_k + 16 * il0;
        const int g = k_abs / grp;                 // 组号
        const int p = k_abs % grp;                 // 组内起点（0/16/32/48）
        device const uchar *gp = wrow + (size_t)g * 36;

        const float scale = (float)(*(device const half *)(gp + 0));
        const float zero  = (float)(*(device const half *)(gp + 2));

        // 2 个 uint 读 8 字节 = 16 个 nibble（4 字节对齐恒成立，见上面的对齐账）
        device const uint *pk = (device const uint *)(gp + 4 + p / 2);
        const uint u0 = pk[0];
        const uint u1 = pk[1];

        float wv[16];
        // u0 = 前 4 字节 = 元素 0..7，u1 = 后 4 字节 = 元素 8..15。
        // 小端序下 u0 的 nibble n 就是元素 n（byte j 的低 nibble = 元素 2j、
        // 高 nibble = 元素 2j+1，所以 nibble 序号与元素序号一致）。
        // 注意不能写成 wv[2j]/wv[2j+1] —— 那会把 u0 的元素撒到偶数位、
        // u1 的撒到奇数位，顺序全错（这正是最初那版误差 4~7 的来源）。
        for (short j = 0; j < 8; ++j) {
            wv[j]     = ((float)((u0 >> (4 * j)) & 0xFu) - zero) * scale;
            wv[j + 8] = ((float)((u1 >> (4 * j)) & 0xFu) - zero) * scale;
        }

        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (short)((tiitg / 2) / 8);
            const short lx = (short)((tiitg / 2) % 8);
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            *(sa + 64 * ib + 8 * ly + lx) = wv[i];
        }

        // ---- PHASE 1b: 暂存激活 tile ----
        {
            const short sx = (short)(tiitg % 4);
            const short sy = (short)((tiitg / 4) / 8);
            const short ly = (short)((tiitg / 4) % 8);
            const short ib = 4 * sx + sy;

            const int k = loop_k + iy;
            threadgroup float *dstp = sb + 64 * ib + 8 * ly;
            if (k + 8 <= in_cnt) {
                *(threadgroup float4 *)(dstp + 0) = *(device const float4 *)(xr + loop_k + 0);
                *(threadgroup float4 *)(dstp + 4) = *(device const float4 *)(xr + loop_k + 4);
            } else {
                for (short i = 0; i < 8; ++i)
                    dstp[i] = (k + i < in_cnt) ? xr[loop_k + i] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- PHASE 2: simdgroup 外积累加（与 f32 变体完全一致）----
        threadgroup const float *lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float *lsmb = sb + 2 * 64 * (sgitg / 2);

        for (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i)
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);

            lsma += 8 * 64;
            lsmb += 4 * 64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *C = Y + (r0 + 32 * (sgitg & 1)) + (size_t)(r1 + 16 * (sgitg >> 1)) * y_stride;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], C + 8 * (i % 4) + (size_t)8 * y_stride * (i / 4), y_stride, 0, false);
    }
}

// ---------------------------------------------------------------------------
// INT4 GEMV：Y[1,out] = X[1,in] @ dequant(W[out,in])^T
//
// 这是 int4 **真正该赢**的形状，与层投影 GEMM 性质完全不同：
//   1. N=1 → 算术强度极低，**带宽受限**（层投影 GEMM 在 n>=64 是算力受限，
//      int4 的带宽红利用不上，所以那边 int4 慢 13-17%）；
//   2. out 巨大（lm_head 151936）→ 线程组数量充裕，**占用率不是问题**
//      （层投影在 n<=32 时只有 96 个线程组、4.8 波、256 线程/核，占用率崩）；
//   3. 一线程一行 → **没有 tile 粒度浪费**（层投影 NR1=32 而 n=8 时 75% 算力白算）；
//   4. 权重只读一遍 → **没有重复反量化**（层投影 gx=16 时同一 tile 反量化 16 次，
//      L2 能吸收重复内存读但吸收不了重复计算 —— 那是层投影 int4 输的根因）。
// MPS 没有 int4 路径，所以这里自写 kernel 是必需的。
//
// 设计同 gemv_wt（f16 版）：一线程一输出行、x 协作载入片上广播复用。
// 差别只在权重读取：每组 36 字节 [scale_fp16|zero_fp16|packed_uint4(32B)]，
// 32 字节 = 8 个 uint = 64 个 nibble。对齐账同 tiled 版 —— 恒定 4 字节对齐，
// 只能读 uint 不能 uint2。
// ---------------------------------------------------------------------------
kernel void gemv_wt_i4(device const uchar *W,
                       device const float *x,
                       device       float *y,
                       constant int &out_cnt, constant int &in_cnt, constant int &grp,
                       constant int &w_packed_stride,
                       threadgroup char *shmem [[threadgroup(0)]],
                       uint tid   [[thread_position_in_grid]],
                       uint tiitg [[thread_index_in_threadgroup]],
                       uint nthr  [[threads_per_threadgroup]]) {
    // 片上分两块：xs = 整个 x 向量（广播复用），xsum = 每组的 Σx_i
    threadgroup float *xs   = (threadgroup float *)(shmem);
    threadgroup float *xsum = (threadgroup float *)(shmem + 4 * (uint)in_cnt);

    const int n_groups = in_cnt / grp;

    for (uint i = tiitg; i < (uint)in_cnt; i += nthr) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Σx_i per group —— **对所有输出行都一样**，所以整组只算一次。
    // 这是下面那个代数变形的关键：把 per-element 的 (q-zero)*scale*x
    // 变成 per-group 的 scale*(Σq*x - zero*Σx)，省掉每元素一次减法+一次乘法。
    for (int g = (int)tiitg; g < n_groups; g += (int)nthr) {
        float s = 0.0f;
        const int base = g * grp;
        for (int i = 0; i < grp; ++i) s += xs[base + i];
        xsum[g] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid >= (uint)out_cnt) return;

    device const uchar *wrow = W + (size_t)tid * w_packed_stride;
    float acc = 0.0f;

    for (int g = 0; g < n_groups; ++g) {
        device const uchar *gp = wrow + (size_t)g * 36;
        const float scale = (float)(*(device const half *)(gp + 0));
        const float zero  = (float)(*(device const half *)(gp + 2));

        // 32 字节 packed = 8 个 uint = 64 个 nibble（低 nibble = 偶数下标元素）
        device const uint *pk = (device const uint *)(gp + 4);
        const int kbase = g * grp;

        // 只累加 Σ q_i·x_i；zero 那一项整组共用 xsum[g]，最后一次性扣掉。
        // per-group ALU 从 ~456 降到 ~196（省掉 64 次减法 + 64 次乘法）。
        float sq = 0.0f;
        for (int u = 0; u < 8; ++u) {
            const uint v = pk[u];
            const int kb = kbase + u * 8;
            sq += (float)((v >>  0) & 0xFu) * xs[kb + 0];
            sq += (float)((v >>  4) & 0xFu) * xs[kb + 1];
            sq += (float)((v >>  8) & 0xFu) * xs[kb + 2];
            sq += (float)((v >> 12) & 0xFu) * xs[kb + 3];
            sq += (float)((v >> 16) & 0xFu) * xs[kb + 4];
            sq += (float)((v >> 20) & 0xFu) * xs[kb + 5];
            sq += (float)((v >> 24) & 0xFu) * xs[kb + 6];
            sq += (float)((v >> 28) & 0xFu) * xs[kb + 7];
        }
        acc += scale * (sq - zero * xsum[g]);
    }

    y[tid] = acc;
}
)MSL";

    // =========================================================================
    // fp16 <-> fp32（与 runtime/metal_prefill.mm 同一套纯位运算实现）
    // =========================================================================
    inline float f16_to_f32(uint16_t h) {
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t man = h & 0x3FFu;
        uint32_t bits;
        if (exp == 0) {
            if (man == 0) {
                bits = sign;
            } else {
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                bits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (man << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    inline uint16_t f32_to_f16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        const uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
        uint32_t man = bits & 0x7FFFFFu;
        if (exp <= 0) return static_cast<uint16_t>(sign);          // 下溢 → ±0
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); // 上溢 → inf
        // 四舍五入到最近（偶数）
        const uint32_t round_bit = 0x1000u;
        man += round_bit + ((man >> 13) & 1u);
        if (man & 0x800000u) { man = 0; exp += 1; }
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (man >> 13));
    }

    inline size_t align16(size_t bytes) { return (bytes + 15u) & ~static_cast<size_t>(15u); }

    // =========================================================================
    // 一个待测 GEMM 形状
    // =========================================================================
    struct Shape {
        const char *name;
        int out_cnt; // M：输出列数（= 权重行数）
        int in_cnt;  // K：reduction 维（= 权重列数）
        int n_cnt;   // N：token 数
    };

    // Qwen3-0.6B @ seq=512 的真实投影形状。
    //   NH=16, NKV=8, HD=128 → QD=2048, KVD=1024
    //   H=1024, I=3072, V=151936
    const Shape kShapes[] = {
        {"qkv     ", 2048 + 2 * 1024, 1024, 512},
        {"o_proj  ", 1024, 2048, 512},
        {"gate_up ", 2 * 3072, 1024, 512},
        {"down    ", 1024, 3072, 512},
        {"lm_head ", 151936, 1024, 512},
    };

    // CPU fp32 参考（只在小规模上跑；大形状用 MPS 当参考）
    void cpu_reference(const std::vector<uint16_t> &W, int out_cnt, int in_cnt,
                       const std::vector<float> &X, int n_cnt, std::vector<float> &Y) {
        Y.assign(static_cast<size_t>(n_cnt) * out_cnt, 0.0f);
        for (int s = 0; s < n_cnt; ++s) {
            for (int o = 0; o < out_cnt; ++o) {
                double acc = 0.0;
                const float *xr = &X[static_cast<size_t>(s) * in_cnt];
                const uint16_t *wr = &W[static_cast<size_t>(o) * in_cnt];
                for (int k = 0; k < in_cnt; ++k) acc += static_cast<double>(xr[k]) * f16_to_f32(wr[k]);
                Y[static_cast<size_t>(s) * out_cnt + o] = static_cast<float>(acc);
            }
        }
    }

    double max_abs_err(const std::vector<float> &a, const std::vector<float> &b) {
        double m = 0.0;
        for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
        return m;
    }

    // =========================================================================
    // --debug：单个线程组的最小形状（out=64, in=32, n=32），打印角落块。
    // 用来定位 swizzle 到底错在哪 —— 值是置换了、还是缩放错了、还是整块偏移。
    // =========================================================================
    int run_debug(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps) {
        const int M = 64, K = 32, N = 32; // 正好 1 个线程组（gx=1, gy=1）

        std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
        std::vector<float> Xh(static_cast<size_t>(N) * K);
        std::mt19937 rng(999);
        std::normal_distribution<float> dw(0.0f, 0.02f);
        std::normal_distribution<float> dx(0.0f, 1.0f);
        for (auto &w : Wh) w = f32_to_f16(dw(rng));
        for (auto &x : Xh) x = dx(rng);

        const int w_stride = K, x_stride = K, y_stride = M; // 紧凑，无 padding

        id<MTLBuffer> bufW = [dev newBufferWithLength:static_cast<size_t>(M) * K * sizeof(uint16_t)
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [dev newBufferWithLength:static_cast<size_t>(N) * K * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [dev newBufferWithLength:static_cast<size_t>(N) * M * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        std::memcpy(bufW.contents, Wh.data(), Wh.size() * sizeof(uint16_t));
        std::memcpy(bufX.contents, Xh.data(), Xh.size() * sizeof(float));

        std::vector<float> ref;
        cpu_reference(Wh, M, K, Xh, N, ref);

        // ---- 自写 kernel ----
        {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps];
            [ce setBuffer:bufW offset:0 atIndex:0];
            [ce setBuffer:bufX offset:0 atIndex:1];
            [ce setBuffer:bufY offset:0 atIndex:2];
            [ce setBytes:&M length:sizeof(int) atIndex:3];
            [ce setBytes:&K length:sizeof(int) atIndex:4];
            [ce setBytes:&N length:sizeof(int) atIndex:5];
            [ce setBytes:&w_stride length:sizeof(int) atIndex:6];
            [ce setBytes:&x_stride length:sizeof(int) atIndex:7];
            [ce setBytes:&y_stride length:sizeof(int) atIndex:8];
            [ce setThreadgroupMemoryLength:kShmemBytes atIndex:0];
            [ce dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        const float *got = static_cast<const float *>(bufY.contents);

        std::printf("=== debug: out=%d in=%d n=%d（单线程组）===\n\n", M, K, N);
        std::printf("Y[n][out] 角落 (n=0..3, out=0..3)\n");
        std::printf("%-6s | %-38s | %-38s\n", "n", "CPU 参考", "自写 kernel");
        for (int s = 0; s < 4; ++s) {
            std::printf("%-6d |", s);
            for (int o = 0; o < 4; ++o) std::printf(" %8.4f", ref[static_cast<size_t>(s) * M + o]);
            std::printf("  |");
            for (int o = 0; o < 4; ++o) std::printf(" %8.4f", got[static_cast<size_t>(s) * M + o]);
            std::printf("\n");
        }

        // 沿 out 方向扫 n=0 那一行，看自写结果是不是"值对位置错"
        std::printf("\nn=0 行，out=0..15：\n  ref: ");
        for (int o = 0; o < 16; ++o) std::printf("%7.3f", ref[o]);
        std::printf("\n  got: ");
        for (int o = 0; o < 16; ++o) std::printf("%7.3f", got[o]);
        std::printf("\n");

        // 沿 n 方向扫 out=0 那一列
        std::printf("\nout=0 列，n=0..15：\n  ref: ");
        for (int s = 0; s < 16; ++s) std::printf("%7.3f", ref[static_cast<size_t>(s) * M]);
        std::printf("\n  got: ");
        for (int s = 0; s < 16; ++s) std::printf("%7.3f", got[static_cast<size_t>(s) * M]);
        std::printf("\n");

        double e = 0.0;
        int worst_s = 0, worst_o = 0;
        for (int s = 0; s < N; ++s)
            for (int o = 0; o < M; ++o) {
                const double d = std::fabs(ref[static_cast<size_t>(s) * M + o] - got[static_cast<size_t>(s) * M + o]);
                if (d > e) { e = d; worst_s = s; worst_o = o; }
            }
        std::printf("\nmax_abs_err=%.4e  最差点 (n=%d, out=%d): ref=%.4f got=%.4f\n",
                    e, worst_s, worst_o, ref[static_cast<size_t>(worst_s) * M + worst_o],
                    got[static_cast<size_t>(worst_s) * M + worst_o]);

        // 全对 / 部分对的格子统计
        int ok = 0, bad = 0;
        for (int s = 0; s < N; ++s)
            for (int o = 0; o < M; ++o) {
                const double d = std::fabs(ref[static_cast<size_t>(s) * M + o] - got[static_cast<size_t>(s) * M + o]);
                if (d < 1e-4) ok++; else bad++;
            }
        std::printf("正确格子 %d / 错误格子 %d（共 %d）\n\n", ok, bad, N * M);
        return bad == 0 ? 0 : 1;
    }

    // =========================================================================
    // --gemv：lm_head 末行形状（out=151936, in=1024, N=1）。
    // MPS N=1 vs 自写 GEMV，并算出实际带宽与屋顶线的差距。
    // =========================================================================
    int run_gemv(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                 int runs) {
        const int M = 151936; // V
        const int K = 1024;   // H

        std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
        std::vector<float> Xh(K);
        std::mt19937 rng(4242);
        std::normal_distribution<float> dw(0.0f, 0.02f);
        std::normal_distribution<float> dx(0.0f, 1.0f);
        for (auto &w : Wh) w = f32_to_f16(dw(rng));
        for (auto &x : Xh) x = dx(rng);

        const int w_stride = static_cast<int>(align16(static_cast<size_t>(K) * sizeof(uint16_t)) / sizeof(uint16_t));
        const size_t w_bytes = static_cast<size_t>(w_stride) * M * sizeof(uint16_t);

        id<MTLBuffer> bufW = [dev newBufferWithLength:w_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [dev newBufferWithLength:K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [dev newBufferWithLength:M * sizeof(float) options:MTLResourceStorageModeShared];

        uint8_t *d = static_cast<uint8_t *>(bufW.contents);
        for (int r = 0; r < M; ++r)
            std::memcpy(d + static_cast<size_t>(r) * w_stride * sizeof(uint16_t),
                        &Wh[static_cast<size_t>(r) * K], static_cast<size_t>(K) * sizeof(uint16_t));
        std::memcpy(bufX.contents, Xh.data(), Xh.size() * sizeof(float));

        // ---- MPS N=1 ----
        MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                       columns:K
                                                                      rowBytes:static_cast<size_t>(w_stride) * sizeof(uint16_t)
                                                                      dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                                       columns:K
                                                                      rowBytes:K * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                                       columns:M
                                                                      rowBytes:M * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bufW descriptor:dW];
        MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
        MPSMatrix *mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                      transposeLeft:NO
                                                                     transposeRight:YES
                                                                         resultRows:1
                                                                      resultColumns:M
                                                                    interiorColumns:K
                                                                              alpha:1.0
                                                                               beta:0.0];

        std::vector<float> y_mps(M), y_gemv(M);

        auto run_mps = [&]() -> double {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mY];
            [cb commit];
            [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        };

        auto run_gemv = [&]() -> double {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps];
            [ce setBuffer:bufW offset:0 atIndex:0];
            [ce setBuffer:bufX offset:0 atIndex:1];
            [ce setBuffer:bufY offset:0 atIndex:2];
            [ce setBytes:&M length:sizeof(int) atIndex:3];
            [ce setBytes:&K length:sizeof(int) atIndex:4];
            [ce setBytes:&w_stride length:sizeof(int) atIndex:5];
            // x 片上暂存：in_cnt 个 float
            [ce setThreadgroupMemoryLength:static_cast<size_t>(K) * sizeof(float) atIndex:0];

            const int tg = static_cast<int>([ps maxTotalThreadsPerThreadgroup]);
            const int groups = (M + tg - 1) / tg;
            [ce dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [ce endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        };

        run_mps(); run_mps();
        std::memcpy(y_mps.data(), bufY.contents, y_mps.size() * sizeof(float));
        run_gemv(); run_gemv();
        std::memcpy(y_gemv.data(), bufY.contents, y_gemv.size() * sizeof(float));

        double best_mps = 1e30, best_gemv = 1e30, worst_mps = 0.0, worst_gemv = 0.0;
        for (int r = 0; r < runs; ++r) {
            double t;
            t = run_mps();  best_mps = std::min(best_mps, t);   worst_mps = std::max(worst_mps, t);
            t = run_gemv(); best_gemv = std::min(best_gemv, t); worst_gemv = std::max(worst_gemv, t);
        }

        const double err = max_abs_err(y_gemv, y_mps);
        const double spread = std::max(worst_mps / best_mps, worst_gemv / best_gemv);

        // 带宽账：至少要读一遍全部 fp16 权重
        const double mb = static_cast<double>(w_bytes) / (1024.0 * 1024.0);
        const double bw_gemv = mb / best_gemv;       // MB/ms = GB/s
        const double bw_mps = mb / best_mps;

        std::printf("=== gemv: lm_head 末行形状 out=%d in=%d N=1 ===\n\n", M, K);
        std::printf("权重量 %.1f MB（fp16）—— 这是必须读的下限\n\n", mb);
        std::printf("%-12s %10s %12s %12s\n", "路径", "GPU ms", "带宽 GB/s", "离散度");
        std::printf("%-12s %10.4f %12.1f %12.2f\n", "MPS  N=1", best_mps, bw_mps, worst_mps / best_mps);
        std::printf("%-12s %10.4f %12.1f %12.2f\n", "自写 GEMV", best_gemv, bw_gemv, worst_gemv / best_gemv);
        std::printf("\n提速 %.2fx    max_abs_err=%.3e    合计离散度 %.2f%s\n",
                    best_mps / best_gemv, err, spread, spread > 1.5 ? "  ⚠️离散，不可用于归因" : "");

        // 绝对正确性：抽 64 行跟 CPU fp64 参考比
        double e_cpu = 0.0;
        for (int o = 0; o < M; o += M / 64) {
            double acc = 0.0;
            const uint16_t *wr = &Wh[static_cast<size_t>(o) * K];
            for (int k = 0; k < K; ++k) acc += static_cast<double>(Xh[k]) * f16_to_f32(wr[k]);
            e_cpu = std::max(e_cpu, std::fabs(acc - static_cast<double>(y_gemv[o])));
        }
        std::printf("vs CPU fp64 参考（抽样 64 行）: %.3e\n", e_cpu);
        return err < 1e-3 ? 0 : 1;
    }

    // =========================================================================
    // int4 量化 + 打包（CPU 侧，与 tools/export_qwen_to_tiny_i4.py 的 RTN 口径一致）
    //
    // 每组 36 字节：[scale_fp16(2B) | zero_fp16(2B) | packed_uint4(32B)]
    // 低 nibble 在前 —— byte j 装元素 2j（低 4 bit）与 2j+1（高 4 bit）。
    // 反量化：(q - zero) × scale。
    // =========================================================================
    void quantize_i4(const std::vector<float> &W, int rows, int cols, int grp,
                     std::vector<uint8_t> &packed, int &packed_stride) {
        const int n_groups = cols / grp;
        packed_stride = n_groups * 36;
        packed.assign(static_cast<size_t>(rows) * packed_stride, 0);

        for (int r = 0; r < rows; ++r) {
            for (int g = 0; g < n_groups; ++g) {
                const float *src = &W[static_cast<size_t>(r) * cols + static_cast<size_t>(g) * grp];
                float mn = src[0], mx = src[0];
                for (int i = 1; i < grp; ++i) { mn = std::min(mn, src[i]); mx = std::max(mx, src[i]); }

                const float scale = (mx - mn) / 15.0f;
                const float zero = (scale > 0.0f) ? std::round(-mn / scale) : 0.0f;

                uint8_t *dst = &packed[static_cast<size_t>(r) * packed_stride + static_cast<size_t>(g) * 36];
                const uint16_t sb = f32_to_f16(scale);
                const uint16_t zb = f32_to_f16(zero);
                std::memcpy(dst + 0, &sb, 2);
                std::memcpy(dst + 2, &zb, 2);

                for (int i = 0; i < grp; i += 2) {
                    const float q0 = std::round(src[i] / scale + zero);
                    const float q1 = std::round(src[i + 1] / scale + zero);
                    const int c0 = std::min(15, std::max(0, static_cast<int>(q0)));
                    const int c1 = std::min(15, std::max(0, static_cast<int>(q1)));
                    dst[4 + i / 2] = static_cast<uint8_t>(c0 | (c1 << 4)); // 低 nibble = 偶数下标
                }
            }
        }
    }

    void dequant_i4_row(const std::vector<uint8_t> &packed, int packed_stride, int cols, int grp,
                        int r, std::vector<float> &out) {
        out.resize(cols);
        const int n_groups = cols / grp;
        for (int g = 0; g < n_groups; ++g) {
            const uint8_t *src = &packed[static_cast<size_t>(r) * packed_stride + static_cast<size_t>(g) * 36];
            uint16_t sb, zb;
            std::memcpy(&sb, src + 0, 2);
            std::memcpy(&zb, src + 2, 2);
            const float scale = f16_to_f32(sb);
            const float zero = f16_to_f32(zb);
            for (int i = 0; i < grp; i += 2) {
                const uint8_t b = src[4 + i / 2];
                out[static_cast<size_t>(g) * grp + i]     = ((float)(b & 0xF) - zero) * scale;
                out[static_cast<size_t>(g) * grp + i + 1] = ((float)(b >> 4) - zero) * scale;
            }
        }
    }

    // =========================================================================
    // --i4：int4 GEMM vs MPS f16 GEMM（同场 A/B）
    //
    // 对照口径：把同一份 int4 权重**反量化成 f16** 喂给 MPS，这样两边算的是同一个
    // 数学问题，差异只来自 kernel 本身。数值参考用 CPU fp64 对反量化后的权重做点积。
    // =========================================================================
    int run_i4(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps_i4,
               int runs) {
        const int GRP = 64;

        std::printf("=== int4 GEMM vs MPS f16（同一份权重，反量化成 f16 喂 MPS）===\n\n");
        std::printf("%-10s %6s %6s %6s | %10s %10s %8s | %10s | %6s\n",
                    "shape", "out", "in", "n", "MPS f16", "int4", "int4/MPS",
                    "int4 err", "spread");
        std::printf("%s\n", std::string(96, '-').c_str());

        double mps_total = 0.0, i4_total = 0.0;

        for (const Shape &sh : kShapes) {
            if (sh.out_cnt > 20000) continue; // 跳过 lm_head，只看四个层投影
            const int M = sh.out_cnt, K = sh.in_cnt, N = sh.n_cnt;

            std::vector<float> Wf(static_cast<size_t>(M) * K);
            std::vector<float> Xh(static_cast<size_t>(N) * K);
            std::mt19937 rng(2024);
            std::normal_distribution<float> dw(0.0f, 0.02f);
            std::normal_distribution<float> dx(0.0f, 1.0f);
            for (auto &w : Wf) w = dw(rng);
            for (auto &x : Xh) x = dx(rng);

            // int4 打包
            std::vector<uint8_t> Wp;
            int w_packed_stride = 0;
            quantize_i4(Wf, M, K, GRP, Wp, w_packed_stride);

            // 同一份权重反量化成 f16，作为 MPS 的输入（保证两边算同一个数学问题）
            std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
            std::vector<float> row;
            for (int r = 0; r < M; ++r) {
                dequant_i4_row(Wp, w_packed_stride, K, GRP, r, row);
                for (int c = 0; c < K; ++c) Wh[static_cast<size_t>(r) * K + c] = f32_to_f16(row[c]);
            }

            const int x_stride = static_cast<int>(align16(static_cast<size_t>(K) * sizeof(float)) / sizeof(float));
            const int y_stride = static_cast<int>(align16(static_cast<size_t>(M) * sizeof(float)) / sizeof(float));

            id<MTLBuffer> bufWp = [dev newBufferWithLength:Wp.size() options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufWf = [dev newBufferWithLength:static_cast<size_t>(M) * K * sizeof(uint16_t)
                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufX = [dev newBufferWithLength:static_cast<size_t>(x_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufY = [dev newBufferWithLength:static_cast<size_t>(y_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            std::memcpy(bufWp.contents, Wp.data(), Wp.size());
            std::memcpy(bufWf.contents, Wh.data(), Wh.size() * sizeof(uint16_t));
            float *xd = static_cast<float *>(bufX.contents);
            for (int r = 0; r < N; ++r)
                std::memcpy(xd + static_cast<size_t>(r) * x_stride, &Xh[static_cast<size_t>(r) * K],
                            static_cast<size_t>(K) * sizeof(float));

            MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                          rowBytes:static_cast<size_t>(K) * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
                                                                          rowBytes:static_cast<size_t>(x_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M
                                                                          rowBytes:static_cast<size_t>(y_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bufWf descriptor:dW];
            MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
            MPSMatrix *mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:N
                resultColumns:M interiorColumns:K alpha:1.0 beta:0.0];

            std::vector<float> y_mps(static_cast<size_t>(N) * y_stride);
            std::vector<float> y_i4(static_cast<size_t>(N) * y_stride);

            auto run_mps = [&]() -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mY];
                [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };
            auto run_i4k = [&]() -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_i4];
                [ce setBuffer:bufWp offset:0 atIndex:0];
                [ce setBuffer:bufX offset:0 atIndex:1];
                [ce setBuffer:bufY offset:0 atIndex:2];
                [ce setBytes:&M length:sizeof(int) atIndex:3];
                [ce setBytes:&K length:sizeof(int) atIndex:4];
                [ce setBytes:&N length:sizeof(int) atIndex:5];
                [ce setBytes:&GRP length:sizeof(int) atIndex:6];
                [ce setBytes:&w_packed_stride length:sizeof(int) atIndex:7];
                [ce setBytes:&x_stride length:sizeof(int) atIndex:8];
                [ce setBytes:&y_stride length:sizeof(int) atIndex:9];
                [ce setThreadgroupMemoryLength:kShmemBytes atIndex:0];
                [ce dispatchThreadgroups:MTLSizeMake((N + kNR1 - 1) / kNR1, (M + kNR0 - 1) / kNR0, 1)
                     threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
                [ce endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };

            run_mps(); run_mps();
            std::memcpy(y_mps.data(), bufY.contents, y_mps.size() * sizeof(float));
            run_i4k(); run_i4k();
            std::memcpy(y_i4.data(), bufY.contents, y_i4.size() * sizeof(float));

            double bm = 1e30, bi = 1e30, wm = 0.0, wi = 0.0;
            for (int r = 0; r < runs; ++r) {
                double t;
                t = run_mps(); bm = std::min(bm, t); wm = std::max(wm, t);
                t = run_i4k(); bi = std::min(bi, t); wi = std::max(wi, t);
            }

            const double err = max_abs_err(y_i4, y_mps);
            const double spread = std::max(wm / bm, wi / bi);
            std::printf("%-10s %6d %6d %6d | %10.4f %10.4f %8.3f | %10.2e | %6.2f%s\n",
                        sh.name, M, K, N, bm, bi, bi / bm, err, spread,
                        spread > 1.5 ? " ⚠️" : "");
            mps_total += bm;
            i4_total += bi;
        }

        std::printf("%s\n", std::string(96, '-').c_str());
        std::printf("%-10s %20s | %10.4f %10.4f %8.3f |\n", "四层合计", "", mps_total, i4_total,
                    i4_total / mps_total);
        std::printf("\n四层合计: MPS f16 %.3f ms → int4 %.3f ms (%.2fx)\n",
                    mps_total, i4_total, mps_total / i4_total);
        std::printf("× 28 层 = MPS %.1f ms vs int4 %.1f ms\n", mps_total * 28, i4_total * 28);
        return 0;
    }

    // =========================================================================
    // --i4-sweep：固定一个 shape，扫 n（token 数），找 int4 的带宽/算力交叉点。
    //
    // 为什么必须扫：seq=512 时层投影 GEMM 是**算力受限**（算术强度 AI=513），
    // int4 的 3.75× 带宽红利根本用不上，所以实测反而慢 12–17%。但短 seq 下
    // 同一个 GEMM 会变成**带宽受限**，那里 int4 才可能赢。
    // 交叉点在哪决定了 int4 到底有没有提速价值 —— 这是本条路线的判据。
    // =========================================================================
    int run_i4_sweep(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps_i4,
                     int runs, int M, int K) {
        const int GRP = 64;
        const int Ns[] = {8, 16, 32, 64, 128, 256, 512};

        std::vector<float> Wf(static_cast<size_t>(M) * K);
        std::mt19937 rng(2024);
        std::normal_distribution<float> dw(0.0f, 0.02f);
        for (auto &w : Wf) w = dw(rng);

        std::vector<uint8_t> Wp;
        int w_packed_stride = 0;
        quantize_i4(Wf, M, K, GRP, Wp, w_packed_stride);

        std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
        std::vector<float> row;
        for (int r = 0; r < M; ++r) {
            dequant_i4_row(Wp, w_packed_stride, K, GRP, r, row);
            for (int c = 0; c < K; ++c) Wh[static_cast<size_t>(r) * K + c] = f32_to_f16(row[c]);
        }

        const double w_i4_mb = static_cast<double>(Wp.size()) / (1024.0 * 1024.0);
        const double w_f16_mb = static_cast<double>(Wh.size()) * 2.0 / (1024.0 * 1024.0);

        std::printf("=== int4 vs MPS f16 扫 n（shape out=%d in=%d）===\n\n", M, K);
        std::printf("权重量: int4 %.2f MB  vs  f16 %.2f MB（%.2fx）\n", w_i4_mb, w_f16_mb, w_f16_mb / w_i4_mb);
        std::printf("带宽地板（权重只读一遍，120 GB/s）: int4 %.3f ms  f16 %.3f ms\n\n",
                    w_i4_mb / 120.0, w_f16_mb / 120.0);

        std::printf("%5s | %10s %10s %9s | %12s %12s | %6s\n",
                    "n", "MPS f16", "int4", "int4/MPS", "f16 GB/s", "int4 GB/s", "spread");
        std::printf("%s\n", std::string(80, '-').c_str());

        id<MTLBuffer> bufWp = [dev newBufferWithLength:Wp.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufWf = [dev newBufferWithLength:Wh.size() * sizeof(uint16_t)
                                              options:MTLResourceStorageModeShared];
        std::memcpy(bufWp.contents, Wp.data(), Wp.size());
        std::memcpy(bufWf.contents, Wh.data(), Wh.size() * sizeof(uint16_t));

        for (int N : Ns) {
            std::vector<float> Xh(static_cast<size_t>(N) * K);
            std::normal_distribution<float> dx(0.0f, 1.0f);
            for (auto &x : Xh) x = dx(rng);

            const int x_stride = static_cast<int>(align16(static_cast<size_t>(K) * sizeof(float)) / sizeof(float));
            const int y_stride = static_cast<int>(align16(static_cast<size_t>(M) * sizeof(float)) / sizeof(float));

            id<MTLBuffer> bufX = [dev newBufferWithLength:static_cast<size_t>(x_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufY = [dev newBufferWithLength:static_cast<size_t>(y_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            float *xd = static_cast<float *>(bufX.contents);
            for (int r = 0; r < N; ++r)
                std::memcpy(xd + static_cast<size_t>(r) * x_stride, &Xh[static_cast<size_t>(r) * K],
                            static_cast<size_t>(K) * sizeof(float));

            MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                          rowBytes:static_cast<size_t>(K) * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K
                                                                          rowBytes:static_cast<size_t>(x_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M
                                                                          rowBytes:static_cast<size_t>(y_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bufWf descriptor:dW];
            MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
            MPSMatrix *mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:N
                resultColumns:M interiorColumns:K alpha:1.0 beta:0.0];

            auto run_mps = [&]() -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mY];
                [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };
            auto run_i4k = [&]() -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps_i4];
                [ce setBuffer:bufWp offset:0 atIndex:0];
                [ce setBuffer:bufX offset:0 atIndex:1];
                [ce setBuffer:bufY offset:0 atIndex:2];
                [ce setBytes:&M length:sizeof(int) atIndex:3];
                [ce setBytes:&K length:sizeof(int) atIndex:4];
                [ce setBytes:&N length:sizeof(int) atIndex:5];
                [ce setBytes:&GRP length:sizeof(int) atIndex:6];
                [ce setBytes:&w_packed_stride length:sizeof(int) atIndex:7];
                [ce setBytes:&x_stride length:sizeof(int) atIndex:8];
                [ce setBytes:&y_stride length:sizeof(int) atIndex:9];
                [ce setThreadgroupMemoryLength:kShmemBytes atIndex:0];
                [ce dispatchThreadgroups:MTLSizeMake((N + kNR1 - 1) / kNR1, (M + kNR0 - 1) / kNR0, 1)
                     threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
                [ce endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };

            run_mps(); run_mps(); run_i4k(); run_i4k();
            double bm = 1e30, bi = 1e30, wm = 0.0, wi = 0.0;
            for (int r = 0; r < runs; ++r) {
                double t;
                t = run_mps(); bm = std::min(bm, t); wm = std::max(wm, t);
                t = run_i4k(); bi = std::min(bi, t); wi = std::max(wi, t);
            }
            const double spread = std::max(wm / bm, wi / bi);
            std::printf("%5d | %10.4f %10.4f %9.3f | %12.1f %12.1f | %6.2f%s\n",
                        N, bm, bi, bi / bm, w_f16_mb / bm, w_i4_mb / bi, spread,
                        spread > 1.5 ? " ⚠️" : (bi < bm ? "  ✅int4 赢" : ""));
        }
        return 0;
    }

    // =========================================================================
    // --i4-gemv：int4 GEMV vs MPS f16 N=1，在 lm_head 形状上。
    //
    // 这是 int4 真正该赢的区间（见 gemv_wt_i4 的注释：带宽受限 + 占用率充裕 +
    // 无 tile 浪费 + 无重复反量化）。同时算实际带宽与屋顶线的差距。
    // =========================================================================
    int run_i4_gemv(id<MTLDevice> dev, id<MTLCommandQueue> queue, id<MTLComputePipelineState> ps,
                    int runs, int M, int K) {
        const int GRP = 64;

        std::vector<float> Wf(static_cast<size_t>(M) * K);
        std::vector<float> Xh(K);
        std::mt19937 rng(4242);
        std::normal_distribution<float> dw(0.0f, 0.02f);
        std::normal_distribution<float> dx(0.0f, 1.0f);
        for (auto &w : Wf) w = dw(rng);
        for (auto &x : Xh) x = dx(rng);

        std::vector<uint8_t> Wp;
        int w_packed_stride = 0;
        quantize_i4(Wf, M, K, GRP, Wp, w_packed_stride);

        std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
        std::vector<float> row;
        for (int r = 0; r < M; ++r) {
            dequant_i4_row(Wp, w_packed_stride, K, GRP, r, row);
            for (int c = 0; c < K; ++c) Wh[static_cast<size_t>(r) * K + c] = f32_to_f16(row[c]);
        }

        id<MTLBuffer> bufWp = [dev newBufferWithLength:Wp.size() options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufWf = [dev newBufferWithLength:Wh.size() * sizeof(uint16_t)
                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [dev newBufferWithLength:K * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [dev newBufferWithLength:M * sizeof(float) options:MTLResourceStorageModeShared];
        std::memcpy(bufWp.contents, Wp.data(), Wp.size());
        std::memcpy(bufWf.contents, Wh.data(), Wh.size() * sizeof(uint16_t));
        std::memcpy(bufX.contents, Xh.data(), Xh.size() * sizeof(float));

        MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                      rowBytes:static_cast<size_t>(K) * sizeof(uint16_t)
                                                                      dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:1 columns:K
                                                                      rowBytes:K * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:1 columns:M
                                                                      rowBytes:M * sizeof(float)
                                                                      dataType:MPSDataTypeFloat32];
        MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bufWf descriptor:dW];
        MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
        MPSMatrix *mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:1
            resultColumns:M interiorColumns:K alpha:1.0 beta:0.0];

        std::vector<float> y_mps(M), y_i4(M);

        auto run_mps = [&]() -> double {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mY];
            [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        };
        auto run_i4k = [&]() -> double {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:ps];
            [ce setBuffer:bufWp offset:0 atIndex:0];
            [ce setBuffer:bufX offset:0 atIndex:1];
            [ce setBuffer:bufY offset:0 atIndex:2];
            [ce setBytes:&M length:sizeof(int) atIndex:3];
            [ce setBytes:&K length:sizeof(int) atIndex:4];
            [ce setBytes:&GRP length:sizeof(int) atIndex:5];
            [ce setBytes:&w_packed_stride length:sizeof(int) atIndex:6];
            // 片上要放 xs[in_cnt] + xsum[in_cnt/grp]
            const size_t shmem_bytes =
                (static_cast<size_t>(K) + static_cast<size_t>(K) / GRP) * sizeof(float);
            [ce setThreadgroupMemoryLength:shmem_bytes atIndex:0];
            const int tg = static_cast<int>([ps maxTotalThreadsPerThreadgroup]);
            [ce dispatchThreadgroups:MTLSizeMake((M + tg - 1) / tg, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [ce endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
        };

        run_mps(); run_mps();
        std::memcpy(y_mps.data(), bufY.contents, y_mps.size() * sizeof(float));
        run_i4k(); run_i4k();
        std::memcpy(y_i4.data(), bufY.contents, y_i4.size() * sizeof(float));

        double bm = 1e30, bi = 1e30, wm = 0.0, wi = 0.0;
        for (int r = 0; r < runs; ++r) {
            double t;
            t = run_mps(); bm = std::min(bm, t); wm = std::max(wm, t);
            t = run_i4k(); bi = std::min(bi, t); wi = std::max(wi, t);
        }

        const double mb_i4 = static_cast<double>(Wp.size()) / (1024.0 * 1024.0);
        const double mb_f16 = static_cast<double>(Wh.size()) * 2.0 / (1024.0 * 1024.0);
        const double err = max_abs_err(y_i4, y_mps);
        const double spread = std::max(wm / bm, wi / bi);

        std::printf("=== int4 GEMV vs MPS f16 N=1（out=%d in=%d）===\n\n", M, K);
        std::printf("%-14s %10s %12s %12s %10s\n", "路径", "权重量 MB", "GPU ms", "带宽 GB/s", "离散度");
        std::printf("%-14s %10.1f %12.4f %12.1f %10.2f\n", "MPS f16 N=1", mb_f16, bm, mb_f16 / bm, wm / bm);
        std::printf("%-14s %10.1f %12.4f %12.1f %10.2f\n", "自写 int4 GEMV", mb_i4, bi, mb_i4 / bi, wi / bi);
        std::printf("\n提速 %.2fx    max_abs_err=%.3e    合计离散度 %.2f%s\n",
                    bm / bi, err, spread, spread > 1.5 ? "  ⚠️离散，不可用于归因" : "");
        std::printf("带宽屋顶线（120 GB/s）: int4 %.3f ms / f16 %.3f ms —— int4 实测离屋顶线 %.1fx\n",
                    mb_i4 / 120.0, mb_f16 / 120.0, bi / (mb_i4 / 120.0));

        double e_cpu = 0.0;
        for (int o = 0; o < M; o += std::max(1, M / 64)) {
            double acc = 0.0;
            dequant_i4_row(Wp, w_packed_stride, K, GRP, o, row);
            for (int k = 0; k < K; ++k) acc += static_cast<double>(Xh[k]) * static_cast<double>(row[k]);
            e_cpu = std::max(e_cpu, std::fabs(acc - static_cast<double>(y_i4[o])));
        }
        std::printf("vs CPU fp64 参考（抽样 64 行）: %.3e\n", e_cpu);
        return 0;
    }

} // namespace

// =============================================================================
// main
// =============================================================================
int main(int argc, char **argv) {
    // 参数：--runs N（默认 7）、--small（附带 CPU 参考绝对校验）
    int runs = 7;
    bool do_small = false;
    bool do_debug = false;
    bool do_gemv = false;
    bool do_i4 = false;
    int sweep_m = 6144, sweep_k = 1024; // 默认 gate_up 形状（四层里最大的）
    bool do_i4_sweep = false;
    bool do_i4_gemv = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--runs") && i + 1 < argc) runs = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--small")) do_small = true;
        else if (!std::strcmp(argv[i], "--debug")) do_debug = true;
        else if (!std::strcmp(argv[i], "--gemv")) do_gemv = true;
        else if (!std::strcmp(argv[i], "--i4")) do_i4 = true;
        else if (!std::strcmp(argv[i], "--i4-sweep")) do_i4_sweep = true;
        else if (!std::strcmp(argv[i], "--i4-gemv")) do_i4_gemv = true;
        else if (!std::strcmp(argv[i], "--shape") && i + 2 < argc) {
            sweep_m = std::atoi(argv[++i]);
            sweep_k = std::atoi(argv[++i]);
        }
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "拿不到 Metal 设备\n"); return 1; }
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        std::printf("设备: %s\n", [[dev name] UTF8String]);
        std::printf("threadgroup memory 上限: %zu bytes（本 kernel 用 %zu）\n",
                    static_cast<size_t>([dev maxThreadgroupMemoryLength]), kShmemBytes);
        std::printf("\n");

        // ---- 编译 shader ----
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:@(kShaderSource) options:nil error:&err];
        if (!lib) {
            std::fprintf(stderr, "MSL 编译失败:\n%s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> fn_f32 = [lib newFunctionWithName:@"gemm_wt_tiled_f32"];
        id<MTLFunction> fn_f16 = [lib newFunctionWithName:@"gemm_wt_tiled_f16"];
        id<MTLFunction> fn_gemv = [lib newFunctionWithName:@"gemv_wt"];
        id<MTLFunction> fn_i4 = [lib newFunctionWithName:@"gemm_wt_tiled_i4"];
        id<MTLFunction> fn_i4_gemv = [lib newFunctionWithName:@"gemv_wt_i4"];
        if (!fn_f32 || !fn_f16 || !fn_gemv || !fn_i4 || !fn_i4_gemv) {
            std::fprintf(stderr, "找不到 kernel 函数\n"); return 1;
        }

        id<MTLComputePipelineState> ps_f32 = [dev newComputePipelineStateWithFunction:fn_f32 error:&err];
        id<MTLComputePipelineState> ps_f16 = [dev newComputePipelineStateWithFunction:fn_f16 error:&err];
        id<MTLComputePipelineState> ps_gemv = [dev newComputePipelineStateWithFunction:fn_gemv error:&err];
        id<MTLComputePipelineState> ps_i4 = [dev newComputePipelineStateWithFunction:fn_i4 error:&err];
        id<MTLComputePipelineState> ps_i4_gemv = [dev newComputePipelineStateWithFunction:fn_i4_gemv error:&err];
        if (!ps_f32 || !ps_f16 || !ps_gemv || !ps_i4 || !ps_i4_gemv) {
            std::fprintf(stderr, "pipeline 创建失败:\n%s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }

        std::printf("maxTotalThreadsPerThreadgroup: f32=%lu f16=%lu gemv=%lu i4=%lu i4_gemv=%lu（tiled 需要 %d）\n\n",
                    static_cast<unsigned long>([ps_f32 maxTotalThreadsPerThreadgroup]),
                    static_cast<unsigned long>([ps_f16 maxTotalThreadsPerThreadgroup]),
                    static_cast<unsigned long>([ps_gemv maxTotalThreadsPerThreadgroup]),
                    static_cast<unsigned long>([ps_i4 maxTotalThreadsPerThreadgroup]),
                    static_cast<unsigned long>([ps_i4_gemv maxTotalThreadsPerThreadgroup]), kThreads);

        if (do_debug) return run_debug(dev, queue, ps_f32);
        if (do_gemv) return run_gemv(dev, queue, ps_gemv, runs);
        if (do_i4) return run_i4(dev, queue, ps_i4, runs);
        if (do_i4_sweep) return run_i4_sweep(dev, queue, ps_i4, runs, sweep_m, sweep_k);
        if (do_i4_gemv) return run_i4_gemv(dev, queue, ps_i4_gemv, runs, sweep_m, sweep_k);

        // ---- 逐个形状 ----
        std::printf("%-10s %6s %6s %6s | %10s %10s %10s | %8s %8s | %8s\n",
                    "shape", "out", "in", "n", "MPS ms", "f32 ms", "f16 ms",
                    "f32 err", "f16 err", "spread");
        std::printf("%s\n", std::string(104, '-').c_str());

        double mps_total = 0.0, f32_total = 0.0, f16_total = 0.0;

        for (const Shape &sh : kShapes) {
            const int M = sh.out_cnt, K = sh.in_cnt, N = sh.n_cnt;

            // 权重 fp16 [M,K]，激活 fp32 [N,K]
            std::vector<uint16_t> Wh(static_cast<size_t>(M) * K);
            std::vector<float> Xh(static_cast<size_t>(N) * K);

            std::mt19937 rng(12345);
            // 权重 ~N(0, 0.02)（典型 Linear 初始化），激活 ~N(0, 1)
            std::normal_distribution<float> dw(0.0f, 0.02f);
            std::normal_distribution<float> dx(0.0f, 1.0f);
            for (auto &w : Wh) w = f32_to_f16(dw(rng));
            for (auto &x : Xh) x = dx(rng);

            // 行跨度（元素）。fp16 权重行跨度按 16 字节对齐 → 元素数为偶数倍数。
            const int w_stride = static_cast<int>(align16(static_cast<size_t>(K) * sizeof(uint16_t)) / sizeof(uint16_t));
            const int x_stride = static_cast<int>(align16(static_cast<size_t>(K) * sizeof(float)) / sizeof(float));
            const int y_stride = static_cast<int>(align16(static_cast<size_t>(M) * sizeof(float)) / sizeof(float));

            id<MTLBuffer> bufW = [dev newBufferWithLength:static_cast<size_t>(w_stride) * M * sizeof(uint16_t)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufX = [dev newBufferWithLength:static_cast<size_t>(x_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bufY = [dev newBufferWithLength:static_cast<size_t>(y_stride) * N * sizeof(float)
                                                 options:MTLResourceStorageModeShared];

            // 逐行拷（处理 stride 与 K 不等的情况）
            {
                uint8_t *d = static_cast<uint8_t *>(bufW.contents);
                for (int r = 0; r < M; ++r)
                    std::memcpy(d + static_cast<size_t>(r) * w_stride * sizeof(uint16_t),
                                &Wh[static_cast<size_t>(r) * K], static_cast<size_t>(K) * sizeof(uint16_t));
                float *dx2 = static_cast<float *>(bufX.contents);
                for (int r = 0; r < N; ++r)
                    std::memcpy(dx2 + static_cast<size_t>(r) * x_stride,
                                &Xh[static_cast<size_t>(r) * K], static_cast<size_t>(K) * sizeof(float));
            }

            // ---- MPS 路径 ----
            // transposeRight:YES —— W 是 [out,in]，要的是 X @ W^T
            MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                           columns:K
                                                                          rowBytes:static_cast<size_t>(w_stride) * sizeof(uint16_t)
                                                                          dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                           columns:K
                                                                          rowBytes:static_cast<size_t>(x_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dY = [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                                           columns:M
                                                                          rowBytes:static_cast<size_t>(y_stride) * sizeof(float)
                                                                          dataType:MPSDataTypeFloat32];
            MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bufW descriptor:dW];
            MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bufX descriptor:dX];
            MPSMatrix *mY = [[MPSMatrix alloc] initWithBuffer:bufY descriptor:dY];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                          transposeLeft:NO
                                                                         transposeRight:YES
                                                                             resultRows:N
                                                                          resultColumns:M
                                                                        interiorColumns:K
                                                                                  alpha:1.0
                                                                                   beta:0.0];

            std::vector<float> y_mps(static_cast<size_t>(N) * y_stride);
            std::vector<float> y_f32(static_cast<size_t>(N) * y_stride);
            std::vector<float> y_f16(static_cast<size_t>(N) * y_stride);

            // ---- 计时闭包：GPU 侧，取 min ----
            auto run_mps = [&]() -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mY];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };

            auto run_custom = [&](id<MTLComputePipelineState> ps) -> double {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:ps];
                [ce setBuffer:bufW offset:0 atIndex:0];
                [ce setBuffer:bufX offset:0 atIndex:1];
                [ce setBuffer:bufY offset:0 atIndex:2];
                [ce setBytes:&M length:sizeof(int) atIndex:3];
                [ce setBytes:&K length:sizeof(int) atIndex:4];
                [ce setBytes:&N length:sizeof(int) atIndex:5];
                [ce setBytes:&w_stride length:sizeof(int) atIndex:6];
                [ce setBytes:&x_stride length:sizeof(int) atIndex:7];
                [ce setBytes:&y_stride length:sizeof(int) atIndex:8];
                [ce setThreadgroupMemoryLength:kShmemBytes atIndex:0];

                const int gx = (N + kNR1 - 1) / kNR1;
                const int gy = (M + kNR0 - 1) / kNR0;
                [ce dispatchThreadgroups:MTLSizeMake(gx, gy, 1)
                     threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
                [ce endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                return (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            };

            // 预热 2 次（MPS kernel 首次 JIT + 页表建立），再取 min
            run_mps(); run_mps();
            std::memcpy(y_mps.data(), bufY.contents, y_mps.size() * sizeof(float));

            run_custom(ps_f32); run_custom(ps_f32);
            std::memcpy(y_f32.data(), bufY.contents, y_f32.size() * sizeof(float));

            run_custom(ps_f16); run_custom(ps_f16);
            std::memcpy(y_f16.data(), bufY.contents, y_f16.size() * sizeof(float));

            double best_mps = 1e30, best_f32 = 1e30, best_f16 = 1e30;
            double worst_mps = 0.0, worst_f32 = 0.0, worst_f16 = 0.0;
            for (int r = 0; r < runs; ++r) {
                double t;
                t = run_mps();          best_mps = std::min(best_mps, t); worst_mps = std::max(worst_mps, t);
                t = run_custom(ps_f32); best_f32 = std::min(best_f32, t); worst_f32 = std::max(worst_f32, t);
                t = run_custom(ps_f16); best_f16 = std::min(best_f16, t); worst_f16 = std::max(worst_f16, t);
            }

            const double err_f32 = max_abs_err(y_f32, y_mps);
            const double err_f16 = max_abs_err(y_f16, y_mps);

            // 离散度取三个变体里最差的（>1.5 不可用于归因）
            const double spread = std::max({worst_mps / best_mps, worst_f32 / best_f32, worst_f16 / best_f16});

            std::printf("%-10s %6d %6d %6d | %10.4f %10.4f %10.4f | %8.2e %8.2e | %8.2f%s\n",
                        sh.name, M, K, N, best_mps, best_f32, best_f16, err_f32, err_f16, spread,
                        spread > 1.5 ? "  ⚠️离散" : "");

            mps_total += best_mps;
            f32_total += best_f32;
            f16_total += best_f16;

            // 可选：小规模 CPU fp32 绝对校验（大形状太慢，只挑小的）
            if (do_small && static_cast<size_t>(M) * K * N < 40000000ull) {
                std::vector<float> ref;
                cpu_reference(Wh, M, K, Xh, N, ref);
                // ref 用紧凑 stride=M，与 y_* 的 y_stride 可能不同，按各自 stride 比对
                double e1 = 0.0, e2 = 0.0;
                for (int s = 0; s < N; ++s)
                    for (int o = 0; o < M; ++o) {
                        const float rv = ref[static_cast<size_t>(s) * M + o];
                        e1 = std::max(e1, static_cast<double>(std::fabs(rv - y_f32[static_cast<size_t>(s) * y_stride + o])));
                        e2 = std::max(e2, static_cast<double>(std::fabs(rv - y_f16[static_cast<size_t>(s) * y_stride + o])));
                    }
                std::printf("           └─ vs CPU fp32: f32=%.2e  f16=%.2e\n", e1, e2);
            }
        }

        std::printf("%s\n", std::string(104, '-').c_str());
        std::printf("%-10s %20s | %10.4f %10.4f %10.4f | %18s | %8s\n",
                    "TOTAL", "", mps_total, f32_total, f16_total, "", "");
        std::printf("\n合计: MPS %.2f ms → f32 %.2f ms (%.2fx) → f16 %.2f ms (%.2fx)\n",
                    mps_total, f32_total, mps_total / f32_total, f16_total, mps_total / f16_total);
    }
    return 0;
}
