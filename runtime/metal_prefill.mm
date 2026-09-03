// ============================================================================
// 文件: metal_prefill.mm
// 作用: Metal GPU prefill 引擎实现 —— 整批 prompt 在 Apple GPU 上做一次前向
//
// 为什么是 .mm 而不是 .cpp?
//   Metal / MetalPerformanceShaders 的接口是 ObjC（id<MTLDevice> 等），
//   必须用 ObjC++ 编译。对外接口在 metal_prefill.h 里是纯 C++，
//   所以 main.cpp 不需要看到任何 ObjC 类型。
//
// 算子分工:
//   - 线性投影（q/k/v/o/gate/up/down/lm_head）: MPS MPSMatrixMultiplication
//     —— GEMM 是 prefill 的绝对主体，交给 MPS 走 Metal 优化路径
//   - RMSNorm / per-head QK norm / RoPE / causal attention / SiLU / 残差:
//     自写 Metal compute kernel（本文件里的 shader 源码字符串，create 时编译）
//   全部算子都在 GPU 上，因此每层只需 1 个 command buffer：把所有 GEMM 与
//   compute dispatch 编码进去，commit 一次、wait 一次。
//   （对比：若每个 GEMM 各开一个 command buffer 并 waitUntilCompleted，
//     0.6B 28 层 × 7 个 GEMM = 196 次 GPU 同步，同步开销直接吃掉 prefill。）
//
//   所有 buffer 都是 MTLResourceStorageModeShared（统一内存），
//   CPU 侧直接用 buf.contents 读写，不需要显式 H2D/D2H 拷贝。
//   仅 embedding lookup 与 KV 写回仍在 CPU（前者按需读一行，后者是输出）。
//
// 数值口径:
//   GEMM 与 compute kernel 都走 fp32（权重 f16→f32 后上传），
//   与 CPU reference 同为 fp32 计算。RoPE 的 cos/sin 表在 CPU 侧按 fp64 算好
//   再上传，避免 GPU fp32 三角函数与 CPU cosf 产生偏差。
// 数值口径:
//   权重 f16 模型直接以 fp16 常驻 GPU，GEMM 走 MPS 的 fp16 矩阵（MPS 内部把
//   fp16 上采成 fp32 再做乘加，所以算术精度与 fp32 路径逐位一致，只是存储与
//   带宽减半）。compute kernel 一律 fp32。RoPE 的 cos/sin 表在 CPU 侧按 fp64
//   算好再上传，避免 GPU fp32 三角函数与 CPU cosf 产生偏差。
//   Qwen3-0.6B 实测末位 logits max_abs_err ≈ 2.8e-5、argmax 与 CPU 一致。
//
// 内存代价:
//   f16 模型权重以 fp16 常驻（0.6B ≈ 1.20 GB），f32 模型以 fp32 常驻（≈ 2.40 GB）。
//   注意 fp16 权重**不提速**：长 prompt prefill 的 GEMM 是算力受限而非带宽受限，
//   砍带宽没有收益；它的价值是把显存减半（0.6B RSS 3.66 GB → 2.46 GB），
//   让更大的模型 / 更长的上下文可行。详见 docs/optimization_log.md。
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "metal_prefill.h"

namespace tinyqwen {

namespace {
    // =========================================================================
    // fp16 -> fp32
    //
    // 纯位运算实现，不依赖 __fp16 硬件指令 —— 这样无论编译到 arm64 还是 x86
    // macOS 都一致。权重是 IEEE754 binary16，这里是标准的 binary16->binary32。
    // =========================================================================
    inline float f16_to_f32(uint16_t h) {
        const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1Fu;
        uint32_t man = h & 0x3FFu;
        uint32_t bits;
        if (exp == 0) {
            if (man == 0) {
                bits = sign; // ±0
            } else {
                // subnormal: 左移规格化，指数相应下调
                exp = 127 - 15 + 1;
                while ((man & 0x400u) == 0) { man <<= 1; exp--; }
                man &= 0x3FFu;
                bits = sign | (exp << 23) | (man << 13);
            }
        } else if (exp == 31) {
            bits = sign | 0x7F800000u | (man << 13); // inf / nan
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        }
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    }

    // =========================================================================
    // GpuMat: 一块 GPU 上的矩阵 + 它在 MPS 眼里的描述
    //
    // row_bytes 向上取整到 16：MPSMatrixDescriptor 要求行跨度 16 字节对齐。
    // 因为 buffer 是 Shared，CPU 侧也要用同一个 row_bytes 做寻址，
    // 否则行数 > 1 时会读到错位的数据。
    // =========================================================================
    struct GpuMat {
        id<MTLBuffer> buf = nil;
        MPSMatrix *m = nil;
        int rows = 0;
        int cols = 0;
        size_t row_bytes = 0;
        size_t elem_size = sizeof(float); // fp32 或 fp16

        float *cpu() const { return static_cast<float *>(buf.contents); }

        float *row(int r) const { return cpu() + static_cast<size_t>(r) * (row_bytes / 4); }

        // fp16 矩阵的行跨度（元素个数）
        int stride_elems() const { return static_cast<int>(row_bytes / elem_size); }
    };

    inline size_t align16(size_t bytes) { return (bytes + 15u) & ~static_cast<size_t>(15u); }

    // 建一个未初始化的激活矩阵（内容由调用方填），一律 fp32。
    // 试过改 fp16 让 MPS 走 fp16 算力路径 —— 实测更慢且误差放大 ~1700×，已回退，
    // 见 docs/optimization_log.md 的证伪记录。
    GpuMat make_activation(id<MTLDevice> dev, int rows, int cols) {
        GpuMat g;
        g.rows = rows;
        g.cols = cols;
        g.row_bytes = align16(static_cast<size_t>(cols) * sizeof(float));
        const size_t bytes = g.row_bytes * static_cast<size_t>(rows);
        g.buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:g.row_bytes
                                                                     dataType:MPSDataTypeFloat32];
        g.m = [[MPSMatrix alloc] initWithBuffer:g.buf descriptor:d];
        return g;
    }

    // 建一个 fp16 权重矩阵：直接把 .tqwen 里的 f16 字节拷进 GPU buffer。
    // 相比 make_weight 省掉整个 f16→f32 转换 —— 显存占用减半（0.6B: 2.38GB → 1.19GB），
    // GEMM 读权重的带宽也减半，create() 也不再需要遍历一遍全模型。
    GpuMat make_weight_f16(id<MTLDevice> dev, const uint16_t *data, int rows, int cols) {
        GpuMat g;
        g.rows = rows;
        g.cols = cols;
        g.elem_size = sizeof(uint16_t);
        g.row_bytes = align16(static_cast<size_t>(cols) * sizeof(uint16_t));
        const size_t bytes = g.row_bytes * static_cast<size_t>(rows);
        g.buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        uint8_t *dst = static_cast<uint8_t *>(g.buf.contents);
        // 逐行拷，处理 row_bytes 与 cols*2 不相等的情况
        for (int r = 0; r < rows; ++r)
            std::memcpy(dst + static_cast<size_t>(r) * g.row_bytes,
                        data + static_cast<size_t>(r) * cols,
                        static_cast<size_t>(cols) * sizeof(uint16_t));
        MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:g.row_bytes
                                                                     dataType:MPSDataTypeFloat16];
        g.m = [[MPSMatrix alloc] initWithBuffer:g.buf descriptor:d];
        return g;
    }

    // 建一个权重矩阵：把 fp32 数据拷进 GPU buffer
    GpuMat make_weight(id<MTLDevice> dev, const float *data, int rows, int cols) {
        GpuMat g;
        g.rows = rows;
        g.cols = cols;
        g.row_bytes = align16(static_cast<size_t>(cols) * sizeof(float));
        const size_t bytes = g.row_bytes * static_cast<size_t>(rows);
        g.buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        float *dst = static_cast<float *>(g.buf.contents);
        // 逐行拷，处理 row_bytes 与 cols*4 不相等的情况
        for (int r = 0; r < rows; ++r)
            std::memcpy(dst + static_cast<size_t>(r) * (g.row_bytes / 4),
                        data + static_cast<size_t>(r) * cols,
                        static_cast<size_t>(cols) * sizeof(float));
        MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:g.row_bytes
                                                                     dataType:MPSDataTypeFloat32];
        g.m = [[MPSMatrix alloc] initWithBuffer:g.buf descriptor:d];
        return g;
    }

    // =========================================================================
    // Metal compute shader 源码
    //
    // 用字符串 + newLibraryWithSource 在 create() 时编译，避免引入 .metal 文件
    // 和额外的构建步骤（metallib）。编译发生在 create()，不在 prefill 计时区内。
    //
    // stride 参数一律是"元素个数"而不是字节：调用方传 row_bytes/4。
    // =========================================================================
    constexpr const char *kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

// 整行 RMSNorm：一个线程一行，行内串行累加平方和。
// 注意：这里刻意不做"线程组内并行归约"——实测更慢（256 线程/行 + 10 次
// threadgroup_barrier 的同步开销，超过它省下的串行遍历时间）。
// GPU 上并行度比 dispatch 次数更值钱，见 docs/optimization_log.md 的证伪记录。
kernel void rmsnorm_rows(device const float *x, device const float *w, device float *y,
                         constant int &rows, constant int &cols, constant int &x_stride,
                         constant int &y_stride, constant float &eps,
                         uint s [[thread_position_in_grid]]) {
    if (s >= (uint)rows) return;
    device const float *xr = x + s * x_stride;
    float ss = 0.0f;
    for (int i = 0; i < cols; ++i) ss += xr[i] * xr[i];
    const float inv = 1.0f / sqrt(ss / (float)cols + eps);
    device float *yr = y + s * y_stride;
    for (int i = 0; i < cols; ++i) yr[i] = xr[i] * inv * w[i];
}

// per-head RMSNorm（Qwen3 稠密的 q_norm/k_norm）：一个线程一个 (行, 头)，原地
kernel void rmsnorm_heads(device float *x, device const float *w,
                          constant int &rows, constant int &heads, constant int &hd,
                          constant int &stride, constant float &eps,
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint)(rows * heads)) return;
    const uint s = tid / (uint)heads;
    const uint h = tid % (uint)heads;
    device float *p = x + s * stride + h * hd;
    float ss = 0.0f;
    for (int i = 0; i < hd; ++i) ss += p[i] * p[i];
    const float inv = 1.0f / sqrt(ss / (float)hd + eps);
    for (int i = 0; i < hd; ++i) p[i] = p[i] * inv * w[i];
}

// RoPE：一个线程一个 (行, 头, 半索引)，前半/后半配对（neoX 风格）。
// 刻意保持"一个线程一对"的最大并行度 —— 与 per-head norm 融合成一个线程
// 会让并行度掉 43×（实测更慢）。
// 参数名用 n_half 而不是 half —— half 是 MSL 的内建类型名（16 位浮点），
// 拿它当参数名会让整个 kernel 解析失败。
kernel void rope_apply(device float *x, device const float *cs, device const float *sn,
                       constant int &rows, constant int &heads, constant int &n_half,
                       constant int &stride, uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint)(rows * heads * n_half)) return;
    const uint i = tid % (uint)n_half;
    const uint h = (tid / (uint)n_half) % (uint)heads;
    const uint s = tid / ((uint)n_half * (uint)heads);
    device float *p = x + s * stride + h * (2 * n_half);
    const float a = p[i];
    const float b = p[i + n_half];
    const float c = cs[s * n_half + i];
    const float v = sn[s * n_half + i];
    p[i] = a * c - b * v;
    p[i + n_half] = b * c + a * v;
}

// causal attention（GQA）：threadgroup 协作版 —— **一个线程组一个 (查询位置, q 头)**。
//
// 为什么必须协作：causal masking 下"一个线程一个 (s1,h)"的负载不均衡与 s1 成正比
// （seq=512 时 s1=0 跑 1 轮、s1=511 跑 512 轮，差 512 倍），GPU 耗时由最长的
// 那些线程决定。
//
// 演进过程（两条弯路都记在 docs/optimization_log.md）：
//   v1 朴素：每线程重读 qp、pass3 直接写 global → seq=512 GPU exec 巨大
//   v2 float4：qp 进寄存器 + float4 + 寄存器累加 → exec min 981 ms
//   v2.5 切块分给多线程 + partial 写 **global** → 全线变慢（占用率崩 + 2GB 流量），已弃
//   v3 本版 threadgroup：中间量全留片上 → exec min **793 ms**（vs v2 提速 1.24×，
//      且离散度更低：1.18 vs 1.35）
//
// 本版关键 —— **所有中间量都留在片上（threadgroup memory）**：
//   1. q 向量由整组协作载入 `qs`（片上），不再占寄存器 —— 消除 qv[32] 的 128 个
//      寄存器，占用率不再受寄存器预算限制；
//   2. 点积分数写 `sc`（片上），不落 global —— 消除 partial 流量；
//   3. 组内按维度分工做加权求和，每线程只负责 hd/nthr 个维度，组内负载均衡。
// threadgroup memory：qs[128] + sc[1024] + red[32] ≈ 4.6KB/组，远小于 32KB 上限。
// 代价是每组 8 次 threadgroup_barrier，但实测净收益为正。
// 要求 hd <= 128 且 max_seq_len <= 1024（片上数组容量）—— create 时校验。
// 把本次的 post-RoPE K/V 从融合 buffer Yqkv 追加进 GPU KV cache。
// 必须在 attention 之前编码进同一个 command buffer —— 否则 attention 读不到新位置。
// 源布局 [n][QD+2*KVD]（q | k | v 三段列切片），目标布局 [nkv][max_seq_len][hd]。
kernel void kv_append(device const float *yqkv, device float *k_cache, device float *v_cache,
                      constant int &n, constant int &pos0, constant int &hd,
                      constant int &yqkv_stride, constant int &k_off, constant int &v_off,
                      constant int &cache_stride,
                      uint2 gid [[thread_position_in_grid]]) {
    // dispatch_2d(enc, ps, cols=NKV*HD, rows=n) —— cols 落在 gid.x，rows 落在 gid.y
    const uint e = gid.x;   // 0..nkv*hd-1
    const uint s = gid.y;   // 本次第几个 token
    if (s >= (uint)n) return;
    const uint kvh = e / (uint)hd;
    const uint i = e % (uint)hd;
    const uint dst = kvh * cache_stride + ((uint)pos0 + s) * (uint)hd + i;
    const uint src = s * yqkv_stride;
    k_cache[dst] = yqkv[src + k_off + e];
    v_cache[dst] = yqkv[src + v_off + e];
}

// causal attention（GQA）：threadgroup 协作版 —— **一个线程组一个 (查询行, q 头)**。
//
// **支持 Q_len != KV_len**（投机解码的 verify pass 形状：K 个草稿 token 去 attend
// 已有的 L 个上下文）。pos0 = 已有上下文长度，本次第 qi 个查询的绝对位置是
// pos0+qi，它能看到 [0, pos0+qi] 共 pos0+qi+1 个 K/V。pos0=0 时退化成普通的
// 对称 causal prefill。
//
// K/V 从 GPU KV cache 读，布局 [n_kv_heads][max_seq_len][head_dim]，
// 所以 cache_stride = max_seq_len * hd（不是行跨度，是"每个 kv 头的整块跨度"）。
//
// 为什么必须协作：causal masking 下"一个线程一个 (s1,h)"的负载不均衡与 s1 成正比
// （seq=512 时 s1=0 跑 1 轮、s1=511 跑 512 轮，差 512 倍），GPU 耗时由最长的
// 那些线程决定。
//
// 本版关键 —— **所有中间量都留在片上（threadgroup memory）**：
//   1. q 向量由整组协作载入 `qs`（片上），不再占寄存器 —— 消除 qv[32] 的 128 个
//      寄存器，占用率不再受寄存器预算限制；
//   2. 点积分数写 `sc`（片上），不落 global —— 消除 partial 流量；
//   3. 组内按维度分工做加权求和，每线程只负责 hd/nthr 个维度，组内负载均衡。
// 占用率账：threadgroup memory ≈ 5KB/组，32KB 上限下每核约并发 6 组；
// 128 线程/组 → ~768 线程/核（实测 32 线程/组只有 ~192，太慢；256 反而更慢）。
// 要求 hd <= 128 且 max_seq_len <= 1024（片上数组容量）—— create 时校验。
kernel void causal_attn(device const float *q, device const float *k_cache,
                        device const float *v_cache, device float *out,
                        constant int &n, constant int &pos0, constant int &nh, constant int &rep,
                        constant int &hd, constant int &q_stride, constant int &cache_stride,
                        constant int &out_stride, constant float &scale,
                        uint2 gid [[threadgroup_position_in_grid]],
                        uint2 tidv [[thread_position_in_threadgroup]],
                        uint2 nthrv [[threads_per_threadgroup]]) {
    threadgroup float qs[128];    // q 向量（片上共享，代替寄存器数组）
    threadgroup float sc[1024];   // attention 分数（片上，代替 global scratch）
    threadgroup float red[128];   // 归约用（容量须 >= kAttnThreads）

    // MSL 要求 kernel 入参要么全标量、要么全同元素数的向量，所以三个位置属性
    // 都声明成 uint2，再取 .x 用
    const uint tid = tidv.x;
    const uint nthr = nthrv.x;
    const uint s1 = gid.x;   // 本次批次内的查询下标
    const uint h = gid.y;
    if (s1 >= (uint)n || h >= (uint)nh) return;
    const uint kvh = h / (uint)rep;
    const uint len = (uint)pos0 + s1 + 1; // causal：能看到 [0, pos0+s1]
    const uint hv = (uint)hd / 4;

    // 1) 协作把 q 载入片上
    device const float *qp = q + s1 * q_stride + h * hd;
    for (uint i = tid; i < (uint)hd; i += nthr) qs[i] = qp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2) 分工算点积，分数写片上（K 用 float4，qs 从片上按标量读）
    device const float *kb = k_cache + kvh * cache_stride;
    const uint chunk = (len + nthr - 1) / nthr;
    const uint lo = tid * chunk;
    const uint hi = min(lo + chunk, len);
    for (uint s2 = lo; s2 < hi; ++s2) {
        device const float4 *kp4 =
            reinterpret_cast<device const float4 *>(kb + s2 * hd);
        float d = 0.0f;
        for (uint j = 0; j < hv; ++j) {
            const float4 kv = kp4[j];
            d += qs[j * 4] * kv.x + qs[j * 4 + 1] * kv.y +
                 qs[j * 4 + 2] * kv.z + qs[j * 4 + 3] * kv.w;
        }
        sc[s2] = d * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3) 归约出全局最大值
    float mx = -1e30f;
    for (uint s2 = tid; s2 < len; s2 += nthr) mx = max(mx, sc[s2]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = max(red[tid], red[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float M = red[0];

    // 4) 算 exp 并写回片上（供第 5 步复用），同时归约出分母
    float sum = 0.0f;
    for (uint s2 = tid; s2 < len; s2 += nthr) {
        const float e = exp(sc[s2] - M);
        sc[s2] = e;
        sum += e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nthr / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];

    // 5) 加权求和：每线程负责一部分维度、遍历全部 s2 —— 组内所有线程工作量相同
    device float *op = out + s1 * out_stride + h * hd;
    device const float *vb = v_cache + kvh * cache_stride;
    for (uint i = tid; i < (uint)hd; i += nthr) {
        float t = 0.0f;
        for (uint s2 = 0; s2 < len; ++s2) t += sc[s2] * vb[s2 * hd + i];
        op[i] = t * inv;
    }
}

// SwiGLU 的 gate * silu(up)：原地写回 g。
// 按行跨度寻址 —— gate 与 up 是融合 buffer Ygu 的两个列切片，不是连续内存。
// 用 2D grid（row, col）而不是 1D + 除模：n*I 达百万级时，每元素一次整数除法
// 是实打实的开销。
kernel void silu_mul(device float *g, device const float *u, constant int &rows,
                     constant int &cols, constant int &stride,
                     uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x;
    const uint row = gid.y;
    if (col >= (uint)cols || row >= (uint)rows) return;
    const uint off = row * stride + col;
    const float x = g[off];
    g[off] = (x / (1.0f + exp(-x))) * u[off];
}

// 残差相加（原地）：要求 dst 与 src 的 stride 一致。
// 刻意一个线程一个元素（n*cols 个线程）—— 与 rmsnorm 融合成"一行一个线程组"
// 会丢掉这个并行度，实测更慢。
kernel void add_rows(device float *dst, device const float *src, constant int &rows,
                     constant int &cols, constant int &stride,
                     uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint)(rows * cols)) return;
    const uint off = (tid / (uint)cols) * stride + (tid % (uint)cols);
    dst[off] += src[off];
}
)MSL";

    // 普通 GPU buffer（不是矩阵，不需要 MPSMatrixDescriptor）
    struct GpuBuf {
        id<MTLBuffer> buf = nil;
        size_t count = 0; // 元素个数

        float *cpu() const { return static_cast<float *>(buf.contents); }
    };

    GpuBuf make_buffer(id<MTLDevice> dev, size_t count) {
        GpuBuf b;
        b.count = count;
        b.buf = [dev newBufferWithLength:count * sizeof(float)
                                 options:MTLResourceStorageModeShared];
        return b;
    }

    // 建一个 buffer 并把 fp32 数据拷进去
    GpuBuf upload_buffer(id<MTLDevice> dev, const float *data, size_t count) {
        GpuBuf b = make_buffer(dev, count);
        std::memcpy(b.cpu(), data, count * sizeof(float));
        return b;
    }

    // attention 线程组宽度。
    // 占用率账：threadgroup memory ≈ 4.6KB/组（qs[128]+sc[1024]+red[128]），
    // 32KB 上限下每核约并发 6 组 —— 32 线程/组只有 ~192 线程/核，太低；
    // 128 线程/组升到 ~768 线程/核。同时 step5 的串行链长度不变但每线程只管
    // hd/128 = 1 个维度，ILP 更好。
    constexpr int kAttnThreads = 128;

    // 1D grid dispatch：按 kernel 的线程组上限向上取整
    void dispatch_1d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int count) {
        const int tg = static_cast<int>([ps maxTotalThreadsPerThreadgroup]);
        const int groups = (count + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    // 线程组协作 dispatch：一个线程组一个 (行, 头)
    void dispatch_groups(id<MTLComputeCommandEncoder> enc, int groups_x, int groups_y) {
        [enc dispatchThreadgroups:MTLSizeMake(groups_x, groups_y, 1)
              threadsPerThreadgroup:MTLSizeMake(kAttnThreads, 1, 1)];
    }

    // 2D grid dispatch：x 方向 cols、y 方向 rows，避免 kernel 内做除模
    void dispatch_2d(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> ps, int cols,
                     int rows) {
        const int tg = static_cast<int>([ps maxTotalThreadsPerThreadgroup]);
        const int groups_x = (cols + tg - 1) / tg;
        [enc dispatchThreadgroups:MTLSizeMake(groups_x, rows, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    }

    // =========================================================================
    // GEMM: Y[seq,out] = X[seq,in] @ W[out,in]^T
    //
    // transposeRight:YES —— HF 权重是 [out,in]，我们要的是 X @ W^T。
    // =========================================================================

    // MPSMatrixMultiplication 的 resultRows/resultColumns/interiorColumns 在建对象时
    // 就固定了，所以按尺寸建。建对象本身有可观开销 —— 必须复用，见 run()。
    MPSMatrixMultiplication *make_mm(id<MTLDevice> dev, int rows, int cols, int interior) {
        return [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                transposeLeft:NO
                                               transposeRight:YES
                                                   resultRows:rows
                                                resultColumns:cols
                                              interiorColumns:interior
                                                        alpha:1.0
                                                         beta:0.0];
    }

    // 只编码不 commit：调用方把整层的 GEMM 与 compute kernel 攒进同一个
    // command buffer，最后统一 commit + wait 一次。
    void gemm_wt_encode(MPSMatrixMultiplication *mm, id<MTLCommandBuffer> cb, const GpuMat &X,
                        const GpuMat &W, const GpuMat &Y) {
        [mm encodeToCommandBuffer:cb leftMatrix:X.m rightMatrix:W.m resultMatrix:Y.m];
    }

    // =========================================================================
    // RoPE cos/sin 表：只依赖位置，与层无关，所以整个 prefill 只算一次
    // （旧实现每层每头都重算一遍 cos/sin，是纯浪费）。
    // 在 CPU 侧按 fp64 算好再上传，避免 GPU fp32 三角函数与 CPU cosf 产生偏差。
    // =========================================================================
    // RoPE cos/sin 表。
    // pos0 是本次批次第一个 token 的**绝对位置** —— 接续调用（投机解码的 verify
    // pass）时 token 落在 [pos0, pos0+n)，不是 [0, n)。表里仍按批次内下标 0..n-1
    // 存放，但角度用绝对位置算，这样 rope_apply 用批次内下标索引就能拿到正确角度。
    void rope_tables(int n, int pos0, int half, float theta, std::vector<float> &cs,
                     std::vector<float> &sn) {
        cs.resize(static_cast<size_t>(n) * half);
        sn.resize(static_cast<size_t>(n) * half);
        for (int s = 0; s < n; ++s) {
            for (int i = 0; i < half; ++i) {
                const double f = static_cast<double>(pos0 + s) /
                                 std::pow(static_cast<double>(theta),
                                          static_cast<double>(i) / half);
                cs[static_cast<size_t>(s) * half + i] = std::cosf(static_cast<float>(f));
                sn[static_cast<size_t>(s) * half + i] = std::sinf(static_cast<float>(f));
            }
        }
    }

    // =========================================================================
    // 权重读取：按 dtype（f16/f32）把 tensor 转成 fp32 连续数组
    // =========================================================================
    // 取 f16 权重的原始字节指针（零拷贝）：.tqwen 里本来就是 f16，
    // 直接指向 ModelFile 的内存即可，不需要转成 fp32。
    // dtype 不是 kF16 时返回 nullptr，调用方应退回 fp32 路径。
    const uint16_t *read_weight_f16_raw(const ModelFile &file, const std::string &name, int rows,
                                        int cols, std::string *err) {
        const TensorView *t = file.get(name);
        if (!t) {
            *err = "missing tensor: " + name;
            return nullptr;
        }
        const size_t expect = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (t->numel() != expect) {
            *err = "shape mismatch for " + name + ": got " + std::to_string(t->numel()) +
                   " expect " + std::to_string(expect);
            return nullptr;
        }
        if (t->dtype != Dtype::kF16) return nullptr; // 交给调用方走 fp32 路径
        return reinterpret_cast<const uint16_t *>(t->data);
    }

    bool read_weight_f32(const ModelFile &file, const std::string &name, int rows, int cols,
                         std::vector<float> &out, std::string *err) {
        const TensorView *t = file.get(name);
        if (!t) {
            *err = "missing tensor: " + name;
            return false;
        }
        const size_t expect = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        if (t->numel() != expect) {
            *err = "shape mismatch for " + name + ": got " + std::to_string(t->numel()) +
                   " expect " + std::to_string(expect);
            return false;
        }
        out.resize(expect);
        if (t->dtype == Dtype::kF32) {
            std::memcpy(out.data(), t->data, expect * sizeof(float));
        } else if (t->dtype == Dtype::kF16) {
            const uint8_t *src = t->data;
            for (size_t i = 0; i < expect; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * sizeof(uint16_t), sizeof(h));
                out[i] = f16_to_f32(h);
            }
        } else {
            *err = "unsupported dtype for " + name + ": " + dtype_name(t->dtype);
            return false;
        }
        return true;
    }

    // 读一行（embedding lookup 用）：避免把整张 embed 表在 CPU 侧常驻
    bool read_row_f32(const ModelFile &file, const std::string &name, int row, int cols,
                      float *out, std::string *err) {
        const TensorView *t = file.get(name);
        if (!t) {
            *err = "missing tensor: " + name;
            return false;
        }
        if (t->dtype == Dtype::kF32) {
            std::memcpy(out, t->f32() + static_cast<size_t>(row) * cols,
                        static_cast<size_t>(cols) * sizeof(float));
        } else if (t->dtype == Dtype::kF16) {
            const uint8_t *src = t->data + static_cast<size_t>(row) * cols * sizeof(uint16_t);
            for (int i = 0; i < cols; ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * sizeof(uint16_t), sizeof(h));
                out[i] = f16_to_f32(h);
            }
        } else {
            *err = "unsupported dtype for " + name + ": " + dtype_name(t->dtype);
            return false;
        }
        return true;
    }

    // 校验模型是否落在本引擎支持的范围内（fail fast，别算出错数还查不出来）
    bool validate_model(const ModelConfig &cfg, std::string *err) {
        if (cfg.full_attention_interval > 1) {
            *err = "Metal prefill 不支持混合架构（GDN）：full_attention_interval=" +
                   std::to_string(cfg.full_attention_interval);
            return false;
        }
        if (cfg.partial_rotary_factor != 1.0f) {
            *err = "Metal prefill 只支持全 RoPE：partial_rotary_factor=" +
                   std::to_string(cfg.partial_rotary_factor);
            return false;
        }
        if (cfg.head_dim == 0 || (cfg.head_dim % 2) != 0) {
            *err = "head_dim 必须是非零偶数，实际 " + std::to_string(cfg.head_dim);
            return false;
        }
        // causal_attn 用 float4 载 K/V/Q，要求 head_dim 能被 4 整除
        if ((cfg.head_dim % 4) != 0) {
            *err = "Metal prefill 的 attention kernel 要求 head_dim % 4 == 0，实际 " +
                   std::to_string(cfg.head_dim);
            return false;
        }
        // 片上数组容量上限（shader 里 threadgroup qs[128]），对应 head_dim <= 128
        if (cfg.head_dim > 128) {
            *err = "Metal prefill 目前支持 head_dim <= 128，实际 " +
                   std::to_string(cfg.head_dim);
            return false;
        }
        if (cfg.n_heads == 0 || cfg.n_kv_heads == 0 || (cfg.n_heads % cfg.n_kv_heads) != 0) {
            *err = "GQA 头数不合法：n_heads=" + std::to_string(cfg.n_heads) +
                   " n_kv_heads=" + std::to_string(cfg.n_kv_heads);
            return false;
        }
        return true;
    }
} // namespace

// =========================================================================
// MetalPrefillEngine: metal_prefill.h 里前向声明的不透明句柄的真实定义
//
// 必须定义在 tinyqwen 命名空间下（而不是匿名命名空间）—— 前向声明是
// tinyqwen::MetalPrefillEngine，若在匿名命名空间里定义就成了另一个类型，
// 所有引用该句柄的函数都会撞上「reference is ambiguous」。
// =========================================================================
struct MetalPrefillEngine {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    std::string dev_name;
    ModelConfig cfg{};
    int max_seq_len = 0;

    // 借来的 ModelFile（embedding lookup 时按需读一行）。
    // 不拥有内存 —— file 必须比引擎活得久，见 metal_prefill.h 的约定。
    const ModelFile *file = nullptr;

    // 每层的权重矩阵（常驻 GPU）。
    // q/k/v 与 gate/up 各自拼成一个大矩阵：把每层 7 次 MPS encode 降到 4 次。
    // 拼接是**替换**而非复制 —— 原来的分散 buffer 不再保留，所以显存中性。
    std::vector<GpuMat> wqkv, wo, wgu, wd;
    // 每层的归一化权重（很小，但要给 GPU kernel 读，所以上传成 buffer）
    std::vector<GpuBuf> ln1, ln2, qn, kn;
    GpuBuf final_norm;

    // lm_head 权重（tied 时就是 embed_tokens）
    GpuMat wlm;
    // embedding 表名（lookup 用）
    std::string embed_name;
    std::string lm_head_name;

    // Metal compute pipeline（create 时从 shader 源码编译一次）
    id<MTLComputePipelineState> ps_rmsnorm_rows = nil;
    id<MTLComputePipelineState> ps_rmsnorm_heads = nil;
    id<MTLComputePipelineState> ps_rope = nil;
    id<MTLComputePipelineState> ps_kv_append = nil;
    id<MTLComputePipelineState> ps_attn = nil;
    id<MTLComputePipelineState> ps_silu = nil;
    id<MTLComputePipelineState> ps_add = nil;

    // GPU 常驻的残差流与辅助 buffer
    GpuMat Hid;      // [max_seq_len, H] 残差流（全程留在 GPU）
    GpuBuf rope_cos; // [max_seq_len, head_dim/2]
    GpuBuf rope_sin; // 同上

    // ---- GPU 常驻 KV cache ----
    // 布局 [n_layers][n_kv_heads][max_seq_len][head_dim]，与 runtime 的 KvCache 一致。
    // 为什么要自己持有一份而不是复用 CPU 的 KvCache：
    //   CPU KvCache 是 std::vector<float>，不是 MTLBuffer，Metal kernel 无法直接访问
    //   （newBufferWithBytesNoCopy 要求页对齐，std::vector 的分配不保证）。
    //   所以这里单独持有一份 GPU 侧缓存 —— 这也是投机解码能跑起来的前提：
    //   verify pass 必须 attend 到已有上下文，而不是从 0 重新算。
    GpuBuf kv_k, kv_v;
    int kv_len = 0;         // 当前已缓存的位置数
    size_t kv_layer_stride = 0; // 单层 K 或 V 的元素数 = n_kv_heads * max_seq_len * head_dim

    // 激活 buffer（按 max_seq_len 分配，run 时只用前 n 行）
    // Xa 与 X 列数不同：o_proj 的输入是 attention 输出 [seq, QD]，
    // 而 X 是 [seq, H] —— 复用 X 会让 GEMM 请求超出矩阵实际列数。
    // Yqkv / Ygu 是融合 GEMM 的输出，q/k/v 与 gate/up 用 buffer offset 切片访问，
    // 不再单独分配 —— 行跨度都是融合矩阵的行跨度。
    GpuMat X, Xa, Yqkv, Yo, Ygu, Yd, Ylm;
    // down_proj 的输入是 Ygu 的前 I 列：共用同一块 buf，只是 columns=I 的
    // 另一个 MPSMatrix 视图（rowBytes 保持 Ygu 的行跨度）
    GpuMat Ygu_gate;
};

// ---------------------------------------------------------------------------
// 平台可用性
// ---------------------------------------------------------------------------
bool metal_prefill_available() { return true; }

std::string metal_prefill_device_name(const MetalPrefillEngine *engine) {
    return engine ? engine->dev_name : std::string();
}

// ---------------------------------------------------------------------------
// KV cache 状态管理
// ---------------------------------------------------------------------------
// reset 只需把长度归零：attention 永远只读 [0, pos0+qi)，旧数据留在 buffer 里
// 也不会被读到，所以不需要真正清零那片内存（省一次 ~235MB 的写）。
void metal_prefill_reset_kv(MetalPrefillEngine *engine) {
    if (engine) engine->kv_len = 0;
}

int metal_prefill_kv_len(const MetalPrefillEngine *engine) {
    return engine ? engine->kv_len : 0;
}

// ---------------------------------------------------------------------------
// create
// ---------------------------------------------------------------------------
bool metal_prefill_create(const ModelFile *file, int max_seq_len, std::string *err,
                          MetalPrefillEngine **out) {
    if (!file || !out) {
        *err = "null argument";
        return false;
    }
    if (!validate_model(file->config(), err)) return false;
    // attention 的片上分数数组容量上限（shader 里 threadgroup sc[1024]）
    if (max_seq_len > 1024) {
        *err = "Metal prefill 目前支持 --max-seq-len <= 1024，实际 " +
               std::to_string(max_seq_len);
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            *err = "no Metal device available";
            return false;
        }

        const ModelConfig &cfg = file->config();
        const int H = static_cast<int>(cfg.hidden_size);
        const int I = static_cast<int>(cfg.intermediate_size);
        const int NH = static_cast<int>(cfg.n_heads);
        const int NKV = static_cast<int>(cfg.n_kv_heads);
        const int HD = static_cast<int>(cfg.head_dim);
        const int V = static_cast<int>(cfg.vocab_size);
        const int L = static_cast<int>(cfg.n_layers);
        const int QD = NH * HD;
        const int KVD = NKV * HD;

        std::string embed_name = "model.embed_tokens.weight";
        std::string lm_head_name = cfg.tied_embeddings ? embed_name : "lm_head.weight";

        // 权重 dtype 只支持 f16 / f32：i4 / vq2 的亚字节布局需要专门的反量化 kernel，
        // 不在本引擎范围内。早报错，别等到读 tensor 时才炸。
        const Dtype dt = static_cast<Dtype>(file->header().dtype);
        if (dt != Dtype::kF32 && dt != Dtype::kF16) {
            *err = std::string("Metal prefill 只支持 f32/f16 权重，实际 ") + dtype_name(dt);
            return false;
        }

        MetalPrefillEngine *e = new MetalPrefillEngine();
        e->dev = dev;
        e->queue = [dev newCommandQueue];
        e->dev_name = std::string(dev.name.UTF8String);
        e->cfg = cfg;
        e->max_seq_len = max_seq_len;
        e->file = file;
        e->embed_name = embed_name;
        e->lm_head_name = lm_head_name;

        // ---- 权重上传 ----
        // 任一 tensor 缺失/形状不符就整体失败并释放已分配的资源，
        // 不做"缺一半权重还继续算"这种静默错误。
        // f16 模型直接零拷贝上传 f16 权重（GEMM 走 fp16，带宽减半）；
        // f32 模型仍走 fp32。两种 dtype 不混用，否则 MPS 会拒绝。
        const bool use_f16 = (dt == Dtype::kF16);
        const bool ok = [&]() -> bool {
            std::vector<float> tmp;
            e->wqkv.resize(L); e->wo.resize(L); e->wgu.resize(L); e->wd.resize(L);
            e->ln1.resize(L); e->ln2.resize(L); e->qn.resize(L); e->kn.resize(L);

            // 优先走 fp16 零拷贝上传；模型不是 f16 dtype 时退回 fp32 转换路径。
            // use_f16 一旦确定就对所有投影权重一致，避免 GEMM 里混用两种 dtype。
            const auto load_w = [&](const std::string &name, int rows, int cols, GpuMat &dst) -> bool {
                if (use_f16) {
                    const uint16_t *raw = read_weight_f16_raw(*file, name, rows, cols, err);
                    if (!raw) return false;
                    dst = make_weight_f16(dev, raw, rows, cols);
                    return true;
                }
                if (!read_weight_f32(*file, name, rows, cols, tmp, err)) return false;
                dst = make_weight(dev, tmp.data(), rows, cols);
                return true;
            };

            // 把多个 [rows_i, cols] 权重按行拼成一个 [Σrows_i, cols] 矩阵。
            // 用于 qkv 三合一 / gate_up 二合一 —— 把每层 7 次 MPS encode 降到 4 次。
            const auto load_concat = [&](const std::vector<std::pair<std::string, int>> &parts,
                                         int cols, GpuMat &dst) -> bool {
                int total = 0;
                for (const auto &p : parts) total += p.second;
                GpuMat g;
                g.rows = total;
                g.cols = cols;
                g.elem_size = use_f16 ? sizeof(uint16_t) : sizeof(float);
                g.row_bytes = align16(static_cast<size_t>(cols) * g.elem_size);
                g.buf = [dev newBufferWithLength:g.row_bytes * static_cast<size_t>(total)
                                         options:MTLResourceStorageModeShared];
                uint8_t *d = static_cast<uint8_t *>(g.buf.contents);
                int r0 = 0;
                for (const auto &p : parts) {
                    if (use_f16) {
                        const uint16_t *raw = read_weight_f16_raw(*file, p.first, p.second, cols, err);
                        if (!raw) return false;
                        for (int r = 0; r < p.second; ++r)
                            std::memcpy(d + static_cast<size_t>(r0 + r) * g.row_bytes,
                                        raw + static_cast<size_t>(r) * cols,
                                        static_cast<size_t>(cols) * sizeof(uint16_t));
                    } else {
                        std::vector<float> t;
                        if (!read_weight_f32(*file, p.first, p.second, cols, t, err)) return false;
                        for (int r = 0; r < p.second; ++r)
                            std::memcpy(d + static_cast<size_t>(r0 + r) * g.row_bytes,
                                        t.data() + static_cast<size_t>(r) * cols,
                                        static_cast<size_t>(cols) * sizeof(float));
                    }
                    r0 += p.second;
                }
                MPSMatrixDescriptor *desc =
                    [MPSMatrixDescriptor matrixDescriptorWithRows:total
                                                         columns:cols
                                                        rowBytes:g.row_bytes
                                                        dataType:use_f16 ? MPSDataTypeFloat16
                                                                         : MPSDataTypeFloat32];
                g.m = [[MPSMatrix alloc] initWithBuffer:g.buf descriptor:desc];
                dst = g;
                return true;
            };

            for (int li = 0; li < L; ++li) {
                const std::string pre = "model.layers." + std::to_string(li) + ".";
                if (!load_concat({{pre + "self_attn.q_proj.weight", QD},
                                  {pre + "self_attn.k_proj.weight", KVD},
                                  {pre + "self_attn.v_proj.weight", KVD}},
                                 H, e->wqkv[li]))
                    return false;
                if (!load_w(pre + "self_attn.o_proj.weight", H, QD, e->wo[li])) return false;
                if (!load_concat({{pre + "mlp.gate_proj.weight", I},
                                  {pre + "mlp.up_proj.weight", I}},
                                 H, e->wgu[li]))
                    return false;
                if (!load_w(pre + "mlp.down_proj.weight", H, I, e->wd[li])) return false;

                if (!read_weight_f32(*file, pre + "input_layernorm.weight", H, 1, tmp, err)) return false;
                e->ln1[li] = upload_buffer(dev, tmp.data(), H);
                if (!read_weight_f32(*file, pre + "post_attention_layernorm.weight", H, 1, tmp, err)) return false;
                e->ln2[li] = upload_buffer(dev, tmp.data(), H);
                // Qwen3 稠密模型才有 per-head q/k norm；缺失时留空表示该层不做
                if (read_weight_f32(*file, pre + "self_attn.q_norm.weight", HD, 1, tmp, err))
                    e->qn[li] = upload_buffer(dev, tmp.data(), HD);
                if (read_weight_f32(*file, pre + "self_attn.k_norm.weight", HD, 1, tmp, err))
                    e->kn[li] = upload_buffer(dev, tmp.data(), HD);
            }

            if (!read_weight_f32(*file, "model.norm.weight", H, 1, tmp, err)) return false;
            e->final_norm = upload_buffer(dev, tmp.data(), H);
            // lm_head 也是 GEMM 权重，同样走 f16 零拷贝（V×H = 622MB → 311MB）
            if (!load_w(lm_head_name, V, H, e->wlm)) return false;
            return true;
        }();

        if (!ok) {
            metal_prefill_destroy(e);
            return false;
        }

        // ---- Metal compute pipeline（shader 源码编译一次）----
        {
            NSError *comp_err = nil;
            NSString *src = [NSString stringWithUTF8String:kShaderSource];
            id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&comp_err];
            if (!lib) {
                *err = "shader 编译失败: " + std::string(comp_err.localizedDescription.UTF8String);
                metal_prefill_destroy(e);
                return false;
            }
            const auto pipeline = [&](const char *name, std::string *perr) {
                NSError *e2 = nil;
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!fn) {
                    *perr = std::string("shader 里没有 kernel ") + name;
                    return (id<MTLComputePipelineState>)nil;
                }
                id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&e2];
                if (!ps) *perr = std::string("pipeline 创建失败 ") + name;
                return ps;
            };
            e->ps_rmsnorm_rows = pipeline("rmsnorm_rows", err);
            e->ps_rmsnorm_heads = pipeline("rmsnorm_heads", err);
            e->ps_rope = pipeline("rope_apply", err);
            e->ps_kv_append = pipeline("kv_append", err);
            e->ps_attn = pipeline("causal_attn", err);
            e->ps_silu = pipeline("silu_mul", err);
            e->ps_add = pipeline("add_rows", err);
            if (!e->ps_rmsnorm_rows || !e->ps_rmsnorm_heads || !e->ps_rope || !e->ps_kv_append ||
                !e->ps_attn || !e->ps_silu || !e->ps_add) {
                metal_prefill_destroy(e);
                return false;
            }
        }

        // ---- 激活 buffer ----
        // 激活 buffer 一律 fp32。
        // 试过把 GEMM 的操作数/结果都改 fp16 让 MPS 走 fp16 算力路径 —— 实测更慢
        // （seq=512 GPU exec 437 → 459 ms）且误差放大 ~1700×，已回退。
        // 原因：MPS 在这些形状上没有从 fp16 算力拿到收益，而所有 compute kernel
        // 的内层循环却多了 half↔float 转换（attention 的点积每 half4 要 4 次
        // static_cast<float>）。见 docs/optimization_log.md 的证伪记录。
        // 权重仍是 fp16（零拷贝上传，省一半显存、零精度代价）。
        e->X = make_activation(dev, max_seq_len, H);
        e->Xa = make_activation(dev, max_seq_len, QD);
        e->Yqkv = make_activation(dev, max_seq_len, QD + 2 * KVD);
        e->Yo = make_activation(dev, max_seq_len, H);
        e->Ygu = make_activation(dev, max_seq_len, 2 * I);
        e->Yd = make_activation(dev, max_seq_len, H);
        e->Ylm = make_activation(dev, max_seq_len, V);
        e->Hid = make_activation(dev, max_seq_len, H);

        // down_proj 的输入 = Ygu 的前 I 列（silu 之后）。同一块 buffer，
        // 只是 columns=I 的另一个 MPSMatrix 视图 —— rowBytes 保持 Ygu 的行跨度。
        e->Ygu_gate.buf = e->Ygu.buf;
        e->Ygu_gate.rows = max_seq_len;
        e->Ygu_gate.cols = I;
        e->Ygu_gate.row_bytes = e->Ygu.row_bytes;
        {
            MPSMatrixDescriptor *d = [MPSMatrixDescriptor matrixDescriptorWithRows:max_seq_len
                                                                          columns:I
                                                                         rowBytes:e->Ygu.row_bytes
                                                                         dataType:MPSDataTypeFloat32];
            e->Ygu_gate.m = [[MPSMatrix alloc] initWithBuffer:e->Ygu.buf descriptor:d];
        }

        // ---- RoPE 表 ----
        // attention 不需要 global scratch：分数走 threadgroup memory（片上）
        e->rope_cos = make_buffer(dev, static_cast<size_t>(max_seq_len) * (HD / 2));
        e->rope_sin = make_buffer(dev, static_cast<size_t>(max_seq_len) * (HD / 2));

        // ---- GPU KV cache ----
        // [n_layers][n_kv_heads][max_seq_len][head_dim]，K/V 各一份。
        // 0.6B @ max_seq_len=1024 约 117MB × 2 = 235MB。
        e->kv_layer_stride = static_cast<size_t>(NKV) * max_seq_len * HD;
        const size_t kv_elems = static_cast<size_t>(L) * e->kv_layer_stride;
        e->kv_k = make_buffer(dev, kv_elems);
        e->kv_v = make_buffer(dev, kv_elems);
        e->kv_len = 0;

        *out = e;
        return true;
    }
}

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------
void metal_prefill_destroy(MetalPrefillEngine *engine) {
    if (!engine) return;
    @autoreleasepool { delete engine; }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------
int metal_prefill_run(MetalPrefillEngine *e, const int *tokens, int n, float *logits_out,
                      KvCache *kv, std::string *err) {
    if (!e || !tokens || n <= 0) {
        *err = "invalid argument";
        return -1;
    }
    if (n > e->max_seq_len) {
        *err = "prompt length " + std::to_string(n) + " exceeds engine capacity " +
               std::to_string(e->max_seq_len);
        return -1;
    }
    // 本次追加在已有上下文之后 —— 这是投机解码 verify pass 的前提。
    // pos0 = 引擎已缓存的位置数；本次的 n 个 token 落在 [pos0, pos0+n)。
    const int pos0 = e->kv_len;
    if (pos0 + n > e->max_seq_len) {
        *err = "已有上下文 " + std::to_string(pos0) + " + 本次 " + std::to_string(n) +
               " 超出引擎容量 " + std::to_string(e->max_seq_len);
        return -1;
    }
    // 若调用方还给了 CPU KvCache，它必须与引擎的 GPU cache 保持同步，
    // 否则 CPU decode 接续会读到错位的历史。
    if (kv && kv->seq_len() != pos0) {
        *err = "CPU KvCache 长度 " + std::to_string(kv->seq_len()) +
               " 与 Metal 引擎的 " + std::to_string(pos0) + " 不一致（需要先 reset 两边）";
        return -1;
    }

    const int H = static_cast<int>(e->cfg.hidden_size);
    const int I = static_cast<int>(e->cfg.intermediate_size);
    const int NH = static_cast<int>(e->cfg.n_heads);
    const int NKV = static_cast<int>(e->cfg.n_kv_heads);
    const int HD = static_cast<int>(e->cfg.head_dim);
    const int V = static_cast<int>(e->cfg.vocab_size);
    const int L = static_cast<int>(e->cfg.n_layers);
    const int QD = NH * HD;
    const int KVD = NKV * HD;
    const int rep = NH / NKV;
    const float eps = e->cfg.rms_norm_eps;
    const float theta = e->cfg.rope_theta;
    const float attn_scale = 1.0f / std::sqrtf(static_cast<float>(HD));

    @autoreleasepool {
        // ---- embedding lookup（唯一还在 CPU 的输入侧工作，按需读一行）----
        for (int s = 0; s < n; ++s) {
            const int tok = tokens[s];
            if (tok < 0 || tok >= V) {
                *err = "token id out of range: " + std::to_string(tok);
                return -1;
            }
            if (!read_row_f32(*e->file, e->embed_name, tok, H,
                              e->Hid.row(s), err))
                return -1;
        }

        // ---- RoPE 表：整个 prefill 只算一次 ----
        {
            std::vector<float> cs, sn;
            rope_tables(n, pos0, HD / 2, theta, cs, sn);
            std::memcpy(e->rope_cos.cpu(), cs.data(), cs.size() * sizeof(float));
            std::memcpy(e->rope_sin.cpu(), sn.data(), sn.size() * sizeof(float));
        }

        // 各 buffer 的行跨度（元素），传给 kernel 做寻址。
        // q/k/v 同住一块 Yqkv buffer，行跨度相同，靠字节 offset 切片。
        // 注意 Hid 是 fp32（残差流保精度），其余 GEMM buffer 是 fp16 ——
        // 跨度按各自的 elem_size 换算，不能一律除 4。
        const int hid_s = static_cast<int>(e->Hid.row_bytes / e->Hid.elem_size);
        const int x_s = static_cast<int>(e->X.row_bytes / e->X.elem_size);
        const int q_s = static_cast<int>(e->Yqkv.row_bytes / e->Yqkv.elem_size);
        const int kv_s = q_s;
        const int xa_s = static_cast<int>(e->Xa.row_bytes / e->Xa.elem_size);
        // 融合 buffer 内的切片起点（字节）。Metal 要求 offset 4 字节对齐，
        // fp32 元素下 QD/KVD/I 乘 4 后成立。
        const NSUInteger off_k = static_cast<NSUInteger>(QD) * sizeof(float);
        const NSUInteger off_v = static_cast<NSUInteger>(QD + KVD) * sizeof(float);
        const NSUInteger off_u = static_cast<NSUInteger>(I) * sizeof(float);

        const auto set_i = [](id<MTLComputeCommandEncoder> ce, int v, NSUInteger idx) {
            [ce setBytes:&v length:sizeof(int) atIndex:idx];
        };
        const auto set_f = [](id<MTLComputeCommandEncoder> ce, float v, NSUInteger idx) {
            [ce setBytes:&v length:sizeof(float) atIndex:idx];
        };

        // ---- MPS 算子：整个 run 只建一次，28 层复用 ----
        // 每层的 GEMM 尺寸完全相同，若每次 GEMM 都新建对象，一次 prefill 就是
        // 28 层 × 7 个 GEMM = 196 次 alloc，这部分开销会直接吃掉 prefill 时间。
        MPSMatrixMultiplication *mm_qkv = make_mm(e->dev, n, QD + 2 * KVD, H);
        MPSMatrixMultiplication *mm_o = make_mm(e->dev, n, H, QD);
        MPSMatrixMultiplication *mm_gu = make_mm(e->dev, n, 2 * I, H);
        MPSMatrixMultiplication *mm_d = make_mm(e->dev, n, H, I);
        MPSMatrixMultiplication *mm_lm = make_mm(e->dev, n, V, H);

        // ---- 可选计时：分离"CPU 侧编码耗时"与"GPU 侧执行耗时" ----
        // TINYQWEN_METAL_TIMING=1 开启。用于定位开销到底在 host 还是 device，
        // 不开时这两个计时器完全不参与（零开销）。
        const bool timing = std::getenv("TINYQWEN_METAL_TIMING") != nullptr;
        double t_enc_ms = 0.0, t_exec_ms = 0.0;
        using Clock = std::chrono::steady_clock;

        // ---- 逐层前向 ----
        // 每层一个 command buffer：所有 GEMM 与 compute kernel 编码进去，
        // 只 commit + wait 一次。MPS 的 encodeToCommandBuffer 会自建 encoder，
        // 所以手动建的 compute encoder 必须先 endEncoding 再交给 MPS。
        for (int li = 0; li < L; ++li) {
            const Clock::time_point t0 = Clock::now();
            id<MTLCommandBuffer> cb = [e->queue commandBuffer];

            // 1) input layernorm：Hid -> X
            {
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:e->ps_rmsnorm_rows];
                [ce setBuffer:e->Hid.buf offset:0 atIndex:0];
                [ce setBuffer:e->ln1[li].buf offset:0 atIndex:1];
                [ce setBuffer:e->X.buf offset:0 atIndex:2];
                set_i(ce, n, 3); set_i(ce, H, 4);
                set_i(ce, hid_s, 5); set_i(ce, x_s, 6);
                set_f(ce, eps, 7);
                dispatch_1d(ce, e->ps_rmsnorm_rows, n);
                [ce endEncoding];
            }

            // 2) q/k/v 投影
            // q/k/v 三合一：一次 GEMM 产出 [n, QD+2*KVD]，后续按 offset 切片
            gemm_wt_encode(mm_qkv, cb, e->X, e->wqkv[li], e->Yqkv);

            // 3) per-head q/k norm + RoPE + causal attention
            //    三者之间没有 MPS 打断，所以共用一个 compute encoder。
            //    刻意不融合成一个 kernel：融合会把 RoPE 的并行度从
            //    n*NH*half 降到 n*(NH+NKV)（seq=512 时 524288 → 12288 线程），
            //    实测比省下的 dispatch 开销更亏。
            {
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                const int half = HD / 2;

                if (e->qn[li].buf) {
                    [ce setComputePipelineState:e->ps_rmsnorm_heads];
                    [ce setBuffer:e->Yqkv.buf offset:0 atIndex:0];
                    [ce setBuffer:e->qn[li].buf offset:0 atIndex:1];
                    set_i(ce, n, 2); set_i(ce, NH, 3); set_i(ce, HD, 4);
                    set_i(ce, q_s, 5); set_f(ce, eps, 6);
                    dispatch_1d(ce, e->ps_rmsnorm_heads, n * NH);
                }
                if (e->kn[li].buf) {
                    [ce setComputePipelineState:e->ps_rmsnorm_heads];
                    [ce setBuffer:e->Yqkv.buf offset:off_k atIndex:0];
                    [ce setBuffer:e->kn[li].buf offset:0 atIndex:1];
                    set_i(ce, n, 2); set_i(ce, NKV, 3); set_i(ce, HD, 4);
                    set_i(ce, kv_s, 5); set_f(ce, eps, 6);
                    dispatch_1d(ce, e->ps_rmsnorm_heads, n * NKV);
                }

                [ce setComputePipelineState:e->ps_rope];
                [ce setBuffer:e->rope_cos.buf offset:0 atIndex:1];
                [ce setBuffer:e->rope_sin.buf offset:0 atIndex:2];
                set_i(ce, n, 3); set_i(ce, half, 5);
                [ce setBuffer:e->Yqkv.buf offset:0 atIndex:0];
                set_i(ce, NH, 4); set_i(ce, q_s, 6);
                dispatch_1d(ce, e->ps_rope, n * NH * half);
                [ce setBuffer:e->Yqkv.buf offset:off_k atIndex:0];
                set_i(ce, NKV, 4); set_i(ce, kv_s, 6);
                dispatch_1d(ce, e->ps_rope, n * NKV * half);

                // 3b) 把 post-RoPE 的 K/V 追加进 GPU KV cache。
                //     必须排在 attention 之前、且在同一个 command buffer 内 ——
                //     encoder 顺序即执行顺序，这样 attention 才能读到新位置。
                const NSUInteger kv_off = static_cast<NSUInteger>(li) * e->kv_layer_stride *
                                          sizeof(float);
                const int cache_stride = static_cast<int>(e->max_seq_len * HD);
                [ce setComputePipelineState:e->ps_kv_append];
                [ce setBuffer:e->Yqkv.buf offset:0 atIndex:0];
                [ce setBuffer:e->kv_k.buf offset:kv_off atIndex:1];
                [ce setBuffer:e->kv_v.buf offset:kv_off atIndex:2];
                set_i(ce, n, 3); set_i(ce, pos0, 4); set_i(ce, HD, 5);
                set_i(ce, kv_s, 6); set_i(ce, QD, 7); set_i(ce, QD + KVD, 8);
                set_i(ce, cache_stride, 9);
                dispatch_2d(ce, e->ps_kv_append, NKV * HD, n);

                // 3c) attention：q 来自本次批次，K/V 全部从 cache 读 [0, pos0+qi]
                [ce setComputePipelineState:e->ps_attn];
                [ce setBuffer:e->Yqkv.buf offset:0 atIndex:0];
                [ce setBuffer:e->kv_k.buf offset:kv_off atIndex:1];
                [ce setBuffer:e->kv_v.buf offset:kv_off atIndex:2];
                [ce setBuffer:e->Xa.buf offset:0 atIndex:3];
                set_i(ce, n, 4); set_i(ce, pos0, 5); set_i(ce, NH, 6); set_i(ce, rep, 7);
                set_i(ce, HD, 8); set_i(ce, q_s, 9); set_i(ce, cache_stride, 10);
                set_i(ce, xa_s, 11); set_f(ce, attn_scale, 12);
                dispatch_groups(ce, n, NH);

                [ce endEncoding];
            }

            // 4) o 投影
            gemm_wt_encode(mm_o, cb, e->Xa, e->wo[li], e->Yo);

            // 5) 残差 + post-attention layernorm：Hid += Yo，Hid -> X
            //    残差刻意用 add_rows 一个线程一个元素（n*H 个线程）；
            //    与 rmsnorm 融合成线程组版会丢掉这个并行度，实测更慢。
            {
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:e->ps_add];
                [ce setBuffer:e->Hid.buf offset:0 atIndex:0];
                [ce setBuffer:e->Yo.buf offset:0 atIndex:1];
                set_i(ce, n, 2); set_i(ce, H, 3); set_i(ce, hid_s, 4);
                dispatch_1d(ce, e->ps_add, n * H);

                [ce setComputePipelineState:e->ps_rmsnorm_rows];
                [ce setBuffer:e->Hid.buf offset:0 atIndex:0];
                [ce setBuffer:e->ln2[li].buf offset:0 atIndex:1];
                [ce setBuffer:e->X.buf offset:0 atIndex:2];
                set_i(ce, n, 3); set_i(ce, H, 4);
                set_i(ce, hid_s, 5); set_i(ce, x_s, 6);
                set_f(ce, eps, 7);
                dispatch_1d(ce, e->ps_rmsnorm_rows, n);
                [ce endEncoding];
            }

            // 6) SwiGLU FFN：gate/up 二合一 GEMM，silu 按行跨度切片处理
            gemm_wt_encode(mm_gu, cb, e->X, e->wgu[li], e->Ygu);
            {
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                const int gu_s = static_cast<int>(e->Ygu.row_bytes / e->Ygu.elem_size);
                [ce setComputePipelineState:e->ps_silu];
                [ce setBuffer:e->Ygu.buf offset:0 atIndex:0];
                [ce setBuffer:e->Ygu.buf offset:off_u atIndex:1];
                set_i(ce, n, 2); set_i(ce, I, 3); set_i(ce, gu_s, 4);
                dispatch_2d(ce, e->ps_silu, I, n);
                [ce endEncoding];
            }
            gemm_wt_encode(mm_d, cb, e->Ygu_gate, e->wd[li], e->Yd);

            // 7) 残差：Hid += Yd
            {
                id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
                [ce setComputePipelineState:e->ps_add];
                [ce setBuffer:e->Hid.buf offset:0 atIndex:0];
                [ce setBuffer:e->Yd.buf offset:0 atIndex:1];
                set_i(ce, n, 2); set_i(ce, H, 3); set_i(ce, hid_s, 4);
                dispatch_1d(ce, e->ps_add, n * H);
                [ce endEncoding];
            }

            const Clock::time_point t1 = Clock::now();
            [cb commit];
            [cb waitUntilCompleted];
            const Clock::time_point t2 = Clock::now();
            if (timing) {
                t_enc_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
                t_exec_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            }
            // ---- KV 写回（post-RoPE），供后续 CPU decode 接续 ----
            if (kv) {
                // k/v 是 Yqkv（fp16）的列切片，起点在 off_k / off_v。
                const float *k = e->Yqkv.cpu() + off_k / sizeof(float);
                const float *v = e->Yqkv.cpu() + off_v / sizeof(float);
                std::vector<float> krow(static_cast<size_t>(KVD));
                std::vector<float> vrow(static_cast<size_t>(KVD));
                for (int s = 0; s < n; ++s) {
                    std::memcpy(krow.data(), k + static_cast<size_t>(s) * kv_s,
                                static_cast<size_t>(KVD) * sizeof(float));
                    std::memcpy(vrow.data(), v + static_cast<size_t>(s) * kv_s,
                                static_cast<size_t>(KVD) * sizeof(float));
                    kv->write_token(li, pos0 + s, krow.data(), vrow.data());
                }
            }
        }

        // ---- final norm + lm_head ----
        {
            id<MTLCommandBuffer> cb = [e->queue commandBuffer];
            id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
            [ce setComputePipelineState:e->ps_rmsnorm_rows];
            [ce setBuffer:e->Hid.buf offset:0 atIndex:0];
            [ce setBuffer:e->final_norm.buf offset:0 atIndex:1];
            [ce setBuffer:e->X.buf offset:0 atIndex:2];
            set_i(ce, n, 3); set_i(ce, H, 4);
            set_i(ce, hid_s, 5); set_i(ce, x_s, 6);
            set_f(ce, eps, 7);
            dispatch_1d(ce, e->ps_rmsnorm_rows, n);
            [ce endEncoding];
            gemm_wt_encode(mm_lm, cb, e->X, e->wlm, e->Ylm);
            [cb commit];
            [cb waitUntilCompleted];
        }

        if (kv) kv->advance(n);
        e->kv_len = pos0 + n; // 引擎侧的 GPU cache 长度同步推进

        // ---- 输出 logits ----
        const size_t lmstride = e->Ylm.row_bytes / e->Ylm.elem_size;
        const float *src = e->Ylm.cpu();
        if (logits_out) {
            for (int s = 0; s < n; ++s)
                std::memcpy(logits_out + static_cast<size_t>(s) * V,
                            src + static_cast<size_t>(s) * lmstride,
                            static_cast<size_t>(V) * sizeof(float));
        }

        // ---- 末位 argmax ----
        const float *last = src + static_cast<size_t>(n - 1) * lmstride;
        int best = 0;
        for (int i = 1; i < V; ++i)
            if (last[i] > last[best]) best = i;

        if (timing) {
            std::fprintf(stderr, "[metal-timing] seq=%d encode=%.3f ms exec=%.3f ms (28 层)\n",
                         n, t_enc_ms, t_exec_ms);
        }
        return best;
    }
}

} // namespace tinyqwen
