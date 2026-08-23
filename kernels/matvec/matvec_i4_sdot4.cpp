// ============================================================================
// matvec_i4_sdot4.cpp — INT4 weight-only matvec：y = W @ x
//                  W4A8 SDOT v4（修正硬件半精度转换）+ 动态调度多线程版
// ============================================================================
// 本文件实现两种变体：单线程 "sdot4" 和多线程 "sdot4_mt"。
//
// 相对 sdot3 的**唯一**改动（数值逐位不变）：修正组头 fp16→fp32 的特性守卫，
// 让硬件转换真正生效。
//
// 背景（sample 指令级采样定位）：单线程 4B decode 97% 时间在
// dot_2rows 内，热点却散布在**软件版 half_to_float** 的分支指令上
// （clz/ubfx/cmp #0x1f/b.eq，每次转换 ~15-20 条指令）——sdot3 的
// `#if defined(__ARM_FEATURE_FP16)` 守卫**从未成立**：clang +fp16 定义的
// 宏是 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC / _VECTOR_ARITHMETIC，
// 根本没有 __ARM_FEATURE_FP16 这个名字。于是每组 4 次转换（两行 ×
// scale+zero）× 65.7M 组/token ≈ 4.6G 条多余指令，是单核效率
// （~15 GB/s，能力 68.8）的第一大头。
//
// 修正：守卫改用 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC，转换用 _Float16
// 标量 cast——编译为 `fmov s,w0` + `fcvt s,h0` 两条指令（软件版
// ~15-20 条 + 分支）。每 32 权重组头开销 ~70 → ~12 条指令。
//
// 其余结构与 sdot3 完全相同：动态行调度（work-stealing）+ 内联 4B 组头
// （零额外流量）+ 128 位解包（vld1q_u8 16B=32 权重 + vzipq + 2×SDOT）。
//
// 布局与 sdot / sdot2 / sdot3 完全一致（每组 [scale_fp16(2B) |
// zero_fp16(2B) | packed_uint4(G/2 B)]）。group_size 必须是 16 的倍数。
//
// 数值不变性（相对 sdot3，逐位一致）：
//   - fp16→fp32 是无损扩展：硬件 FCVT 与软件 half_to_float 输出逐位相同；
//   - 其余路径一行未改。
//
// 条件编译：需要 __aarch64__ && __ARM_FEATURE_DOTPROD；
// 硬件转换路径需 __ARM_FEATURE_FP16_SCALAR_ARITHMETIC（CMake 对本文件
// 开 +fp16）。缺特性时退回软件 half_to_float（仍逐位一致，只是慢一点）。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float（无 FP16 特性时的软件兜底）

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstdio>           // fprintf（超限报错）
#include <cstdlib>          // abort（超限报错）
#include <cstring>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

constexpr int kGroupHeader = 4;    // 每组头部字节数（scale_fp16 + zero_fp16）
// 支持的最大输入维度。与 sdot2 同约束：必须 >= 模型最大 in_dim
// （Qwen3.5-4B 的 down_proj = 9216），取 16384 留余量；超限显式 abort，
// 绝不静默截断（截断 = 高维乘脏数据 → 乱码，4B 曾因此整段乱码）。
constexpr int kMaxInDim = 16384;

// 全局 scratch 缓冲区（matvec 串行调用，静态安全；与 sdot2 各自独立）
int8_t  g_xq4[kMaxInDim];            // int8 量化后的激活
int32_t g_xq4_prefix[kMaxInDim + 1]; // 激活前缀和（O(1) 组内求和）

// ========================================================================
// quantize_x_i8_v3() — 对称 int8 量化 x，并填前缀和
// ========================================================================
// 与 sdot2 的 quantize_x_i8_v2 逐字同逻辑（保证两变体激活完全一致），
// 仅使用本文件独立的 scratch 缓冲区。
float quantize_x_i8_v3(const float *x, int in_dim) {
    // 防御：静态缓冲区上限，超限必须显式报错（理由同 kMaxInDim 注释）
    if (in_dim > kMaxInDim) {
        fprintf(stderr,
                "[matvec_i4_sdot4] fatal: in_dim=%d 超过 kMaxInDim=%d，"
                "请调大 kernels/matvec/matvec_i4_sdot4.cpp 的 kMaxInDim\n",
                in_dim, kMaxInDim);
        abort();
    }
    const int n = in_dim;
    // 第一遍：找绝对值最大值（定对称量化范围）
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    // 缩放因子：把 [-amax, amax] 映到 [-127, 127]
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale_x;
    g_xq4_prefix[0] = 0;
    // 第二遍：逐元素量化（四舍五入远离零）+ 填前缀和
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v); // clamp
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);
        g_xq4[i] = static_cast<int8_t>(q);
        g_xq4_prefix[i + 1] = g_xq4_prefix[i] + q;
    }
    return scale_x;
}

// ========================================================================
// half_bits_to_float() — fp16 位模式 → fp32
// ========================================================================
// 有 FEAT_FP16 时用硬件转换：_Float16 标量 cast 编译为
// `fmov s0, w0` + `fcvt s0, h0`（2 条指令）；否则退回软件实现
// （无损但分支多）。两条路径输出逐位一致：fp16→fp32 本来就是精确扩展。
//
// ⚠️ 守卫宏的坑（sdot3 在此翻车）：clang -march=...+fp16 **不定义**
// __ARM_FEATURE_FP16（这个名字不存在），定义的是
// __ARM_FEATURE_FP16_SCALAR_ARITHMETIC 与 __ARM_FEATURE_FP16_VECTOR_ARITHMETIC。
// sdot3 用错了守卫，热循环静默回退软件转换（每次 ~15-20 条指令 + 分支，
// sample 里的第一大头）。此处改用 _*_SCALAR_ARITHMETIC，并用 #error 之外的
// 兜底策略：缺特性仍可编译（软件路径），不阻断非 +fp16 环境。
inline float half_bits_to_float(uint16_t h) {
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    _Float16 hf;
    std::memcpy(&hf, &h, sizeof(hf));
    return static_cast<float>(hf);
#else
    return half_to_float(h);
#endif
}

// ========================================================================
// dot_row_i4_sdot4() — 单行点积（内联组头 + 128 位解包）
// ========================================================================
// 参数：
//   row        — 该行权重起始地址（含组头）
//   in_dim     — 输入维度
//   group_size — 量化分组大小
//   scale_x    — 激活量化缩放因子
// 返回值：该行输出（fp32）
float dot_row_i4_sdot4(const uint8_t *row, int in_dim, int group_size, float scale_x) {
    const int gpr = (in_dim + group_size - 1) / group_size; // 每行组数
    const int group_bytes = kGroupHeader + group_size / 2;  // 每组字节数
    float acc = 0.0f;   // 行内浮点累加器
    int col = 0;        // 当前组在 x 中的起始列

    const uint8x16_t v8q  = vdupq_n_u8(8);     // 常量 8（128 位，q-8 偏移）
    const uint8x16_t m0Fq = vdupq_n_u8(0x0F);  // 低 nibble 掩码（128 位）
    const uint8x8_t  m0Fd = vdup_n_u8(0x0F);   // 64 位版本（16 元素兜底路）

    for (int g = 0; g < gpr; ++g) {
        // ---- 内联读组头：与 packed 同一缓存行，零额外流量 ----
        const uint8_t *gp = row + static_cast<size_t>(g) * group_bytes;
        uint16_t sh, zh;
        std::memcpy(&sh, gp, 2);      // scale（fp16 位模式）
        std::memcpy(&zh, gp + 2, 2);  // zero（fp16 位模式）
        const float A = half_bits_to_float(sh) * scale_x;        // DOT 系数
        const float C = A * (half_bits_to_float(zh) - 8.0f);     // XQSUM 系数

        const uint8_t *packed = gp + kGroupHeader; // packed nibbles 起始
        const int group_elems = (col + group_size <= in_dim) ? group_size
                                                               : (in_dim - col);
        // ---- 主循环：每次 32 权重（16B packed）----
        int32x4_t acc_dot = vdupq_n_s32(0);
        int i = 0;
        const int n32 = group_elems & ~31; // 32 的整数倍部分
        for (; i < n32; i += 32) {
            // 激活：两段 16B（与权重 32 元素对应）
            int8x16_t xq0 = vld1q_s8(g_xq4 + col + i);
            int8x16_t xq1 = vld1q_s8(g_xq4 + col + i + 16);
            // 权重：16B = 32 个 uint4 → 拆 nibble → 交错还原原始顺序
            uint8x16_t raw = vld1q_u8(packed + i / 2);
            uint8x16_t lo  = vandq_u8(raw, m0Fq);   // 偶数位权重（低 nibble）
            uint8x16_t hi  = vshrq_n_u8(raw, 4);    // 奇数位权重（高 nibble）
            uint8x16x2_t z = vzipq_u8(lo, hi);      // val[0]=w0..w15, val[1]=w16..w31
            int8x16_t q0 = vreinterpretq_s8_u8(vsubq_u8(z.val[0], v8q)); // q-8
            int8x16_t q1 = vreinterpretq_s8_u8(vsubq_u8(z.val[1], v8q));
            // 两条 SDOT 进同一累加器（整数和与顺序无关）
            acc_dot = vdotq_s32(acc_dot, q0, xq0);
            acc_dot = vdotq_s32(acc_dot, q1, xq1);
        }
        // ---- 16 权重兜底块（组内剩余 16..31 元素时）----
        if (i + 16 <= group_elems) {
            uint8x8_t raw = vld1_u8(packed + i / 2);
            uint8x8_t lo = vand_u8(raw, m0Fd);
            uint8x8_t hi = vshr_n_u8(raw, 4);
            uint8x8x2_t z = vzip_u8(lo, hi);
            int8x16_t q = vreinterpretq_s8_u8(
                vsubq_u8(vcombine_u8(z.val[0], z.val[1]), v8q));
            int8x16_t xq = vld1q_s8(g_xq4 + col + i);
            acc_dot = vdotq_s32(acc_dot, q, xq);
            i += 16;
        }
        // ---- 横向归约 + 标量尾部 ----
        int dot_g = vaddvq_s32(acc_dot);
        for (; i < group_elems; ++i) {
            const int byte_idx = i / 2;
            const int q = (i % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                       : ((packed[byte_idx] >> 4) & 0x0F);
            dot_g += (q - 8) * static_cast<int>(g_xq4[col + i]);
        }
        // ---- 浮点还原（与 sdot2 同序）----
        const int xqsum_g = g_xq4_prefix[col + group_elems] - g_xq4_prefix[col];
        acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum_g);
        col += group_size;
    }
    return acc;
}

// ========================================================================
// dot_2rows_i4_sdot4() — 2 行并行点积：共享激活加载 + 独立链
// ========================================================================
// 与 sdot2 的 dot_2rows 同思路：两行各自一条 SDOT 链（ILP），
// 激活只加载一次。解包换成 128 位路径，组头内联读。
void dot_2rows_i4_sdot4(const uint8_t *row_a, const uint8_t *row_b,
                        int in_dim, int group_size, float scale_x,
                        float *out_a, float *out_b) {
    const int gpr = (in_dim + group_size - 1) / group_size;
    const int group_bytes = kGroupHeader + group_size / 2;
    float acc_a = 0.0f, acc_b = 0.0f; // 两行各自的浮点累加器
    int col = 0;

    const uint8x16_t v8q  = vdupq_n_u8(8);
    const uint8x16_t m0Fq = vdupq_n_u8(0x0F);
    const uint8x8_t  m0Fd = vdup_n_u8(0x0F);

    for (int g = 0; g < gpr; ++g) {
        // ---- 两行组头各自内联读 ----
        const uint8_t *gp_a = row_a + static_cast<size_t>(g) * group_bytes;
        const uint8_t *gp_b = row_b + static_cast<size_t>(g) * group_bytes;
        uint16_t sha, zha, shb, zhb;
        std::memcpy(&sha, gp_a, 2);
        std::memcpy(&zha, gp_a + 2, 2);
        std::memcpy(&shb, gp_b, 2);
        std::memcpy(&zhb, gp_b + 2, 2);
        const float A_a = half_bits_to_float(sha) * scale_x;
        const float C_a = A_a * (half_bits_to_float(zha) - 8.0f);
        const float A_b = half_bits_to_float(shb) * scale_x;
        const float C_b = A_b * (half_bits_to_float(zhb) - 8.0f);

        const uint8_t *pk_a = gp_a + kGroupHeader;
        const uint8_t *pk_b = gp_b + kGroupHeader;
        const int group_elems = (col + group_size <= in_dim) ? group_size
                                                               : (in_dim - col);
        // ---- 主循环：32 权重/次，两条独立链 ----
        int32x4_t dot_a = vdupq_n_s32(0);
        int32x4_t dot_b = vdupq_n_s32(0);
        int i = 0;
        const int n32 = group_elems & ~31;
        for (; i < n32; i += 32) {
            // 激活共享：两行复用同两段 16B 加载
            int8x16_t xq0 = vld1q_s8(g_xq4 + col + i);
            int8x16_t xq1 = vld1q_s8(g_xq4 + col + i + 16);

            // 行 A：128 位解包 + 2×SDOT
            uint8x16_t raw_a = vld1q_u8(pk_a + i / 2);
            uint8x16x2_t za = vzipq_u8(vandq_u8(raw_a, m0Fq),
                                       vshrq_n_u8(raw_a, 4));
            dot_a = vdotq_s32(dot_a,
                vreinterpretq_s8_u8(vsubq_u8(za.val[0], v8q)), xq0);
            dot_a = vdotq_s32(dot_a,
                vreinterpretq_s8_u8(vsubq_u8(za.val[1], v8q)), xq1);

            // 行 B：独立链（不依赖 dot_a，提高 ILP）
            uint8x16_t raw_b = vld1q_u8(pk_b + i / 2);
            uint8x16x2_t zb = vzipq_u8(vandq_u8(raw_b, m0Fq),
                                       vshrq_n_u8(raw_b, 4));
            dot_b = vdotq_s32(dot_b,
                vreinterpretq_s8_u8(vsubq_u8(zb.val[0], v8q)), xq0);
            dot_b = vdotq_s32(dot_b,
                vreinterpretq_s8_u8(vsubq_u8(zb.val[1], v8q)), xq1);
        }
        // ---- 16 权重兜底块 ----
        if (i + 16 <= group_elems) {
            uint8x8_t raw_a = vld1_u8(pk_a + i / 2);
            uint8x8x2_t za = vzip_u8(vand_u8(raw_a, m0Fd), vshr_n_u8(raw_a, 4));
            int8x16_t qa = vreinterpretq_s8_u8(
                vsubq_u8(vcombine_u8(za.val[0], za.val[1]), v8q));
            uint8x8_t raw_b = vld1_u8(pk_b + i / 2);
            uint8x8x2_t zb = vzip_u8(vand_u8(raw_b, m0Fd), vshr_n_u8(raw_b, 4));
            int8x16_t qb = vreinterpretq_s8_u8(
                vsubq_u8(vcombine_u8(zb.val[0], zb.val[1]), v8q));
            int8x16_t xq = vld1q_s8(g_xq4 + col + i);
            dot_a = vdotq_s32(dot_a, qa, xq);
            dot_b = vdotq_s32(dot_b, qb, xq);
            i += 16;
        }
        // ---- 归约 + 标量尾部 ----
        int sum_a = vaddvq_s32(dot_a);
        int sum_b = vaddvq_s32(dot_b);
        for (; i < group_elems; ++i) {
            const int bi = i / 2;
            const int qa = (i % 2 == 0) ? (pk_a[bi] & 0x0F) : ((pk_a[bi] >> 4) & 0x0F);
            const int qb = (i % 2 == 0) ? (pk_b[bi] & 0x0F) : ((pk_b[bi] >> 4) & 0x0F);
            const int xq_v = static_cast<int>(g_xq4[col + i]);
            sum_a += (qa - 8) * xq_v;
            sum_b += (qb - 8) * xq_v;
        }
        // ---- 浮点还原（与 sdot2 同序）----
        const int xqsum_g = g_xq4_prefix[col + group_elems] - g_xq4_prefix[col];
        const float xs = static_cast<float>(xqsum_g);
        acc_a += A_a * static_cast<float>(sum_a) - C_a * xs;
        acc_b += A_b * static_cast<float>(sum_b) - C_b * xs;
        col += group_size;
    }
    *out_a = acc_a;
    *out_b = acc_b;
}

// ========================================================================
// matvec_i4_sdot4() — 单线程入口
// ========================================================================
void matvec_i4_sdot4(const uint8_t *w, const float *x, float *y,
                     int out_dim, int in_dim, int group_size) {
    const int gpr = (in_dim + group_size - 1) / group_size;
    const int row_bytes = gpr * (kGroupHeader + group_size / 2);

    // 一次性量化激活（前缀和随路填好）
    const float scale_x = quantize_x_i8_v3(x, in_dim);

    // 2 行并行主循环 + 奇数行兜底（与 sdot2 同结构）
    int o = 0;
    for (; o + 1 < out_dim; o += 2) {
        dot_2rows_i4_sdot4(w + static_cast<size_t>(o) * row_bytes,
                           w + static_cast<size_t>(o + 1) * row_bytes,
                           in_dim, group_size, scale_x, &y[o], &y[o + 1]);
    }
    if (o < out_dim) {
        y[o] = dot_row_i4_sdot4(w + static_cast<size_t>(o) * row_bytes,
                                in_dim, group_size, scale_x);
    }
}

// ========================================================================
// 多线程版 sdot4_mt —— 动态行调度（work-stealing）
// ========================================================================

// 自旋等待（与 sdot2 相同：先 YIELD 再让出）
inline void spin_until_sdot4(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
    int spins = 0;
    while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
#if defined(__aarch64__)
            __builtin_arm_yield();
#endif
        } else {
            std::this_thread::yield();
        }
    }
}

// 默认并行度：硬件并发数（上限 16），TINYQWEN_MT_THREADS 可覆盖。
// 与 sdot2 不同的关键点：这里敢用满全部核，因为动态调度消除了
// 静态切分下慢核拖尾的问题（见文件头 ①）。
inline int default_parallelism_sdot4() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolSdot3 — 常驻线程池 + 原子行计数器动态取块
// ========================================================================
// 对比 sdot2 的 RowPoolSdot2（静态等分 [begin, end)）：
//   - 每次从 next_row 原子地取 chunk_rows 行，处理完再取，直到取完。
//   - 快核（P 簇）自然多取、慢核（E 簇）少取，总时长 ≈ 总工作量 ÷
//     聚合吞吐，而不是被最慢一份静态切分拖住。
//   - 块大小取 out_dim/(4p)（下限 16 行）：每核约 4 块，既摊薄原子
//     开销，又留足再平衡空间。
struct RowPoolSdot3 {
    const uint8_t *w = nullptr;
    float *y = nullptr;
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 64;
    float scale_x = 1.0f;
    int chunk_rows = 16;          // 每次领取的行数（run 时设定）
    int row_stride = 0;           // 行字节数（run 时设定）
    std::atomic<int> next_row{0}; // 下一个未领取的行号

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolSdot3() {
        const int p = default_parallelism_sdot4();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ~RowPoolSdot3() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void worker_main(int) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_sdot4(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_work();                       // 与 master 相同的取块逻辑
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    // 动态取块处理：fetch_add 领一段行区间，2 行并行推进，直到领完。
    // 注意：会修改 next_row 原子计数器，故不能标 const。
    void do_work() {
        for (;;) {
            const int begin = next_row.fetch_add(chunk_rows, std::memory_order_relaxed);
            if (begin >= out_dim) break;     // 行已领完
            const int end = (begin + chunk_rows < out_dim) ? begin + chunk_rows
                                                             : out_dim;
            int o = begin;
            for (; o + 1 < end; o += 2) {
                dot_2rows_i4_sdot4(w + static_cast<size_t>(o) * row_stride,
                                   w + static_cast<size_t>(o + 1) * row_stride,
                                   in_dim, group_size, scale_x,
                                   &y[o], &y[o + 1]);
            }
            if (o < end) {
                y[o] = dot_row_i4_sdot4(w + static_cast<size_t>(o) * row_stride,
                                        in_dim, group_size, scale_x);
            }
        }
    }

    // fork-join 入口
    void run(const uint8_t *w_, float *y_, int out_dim_, int in_dim_,
             int group_size_, float scale_x_) {
        w = w_; y = y_;
        out_dim = out_dim_; in_dim = in_dim_;
        group_size = group_size_; scale_x = scale_x_;
        const int gpr = (in_dim + group_size - 1) / group_size;
        row_stride = gpr * (kGroupHeader + group_size / 2);
        const int p = static_cast<int>(workers.size()) + 1;
        // 块大小：总行 / (4 × 核数)，下限 16 行（太碎则原子开销占比上升）
        chunk_rows = out_dim / (p * 4);
        if (chunk_rows < 16) chunk_rows = 16;
        next_row.store(0, std::memory_order_relaxed);
        // release：上面的字段写入对看到 job_gen 的 worker 全部可见
        job_gen.store(++job_counter, std::memory_order_release);
        do_work();                           // master 同样参与取块
        expected_done += workers.size();
        spin_until_sdot4(done_gen, expected_done);
    }
};

// Meyers singleton：进程内唯一常驻池
RowPoolSdot3 &pool_sdot4() {
    static RowPoolSdot3 p;
    return p;
}

// 粒度阈值：与 sdot2 一致——总元素数低于此值走单线程，
// 避免小矩阵上同步开销吃掉并行收益。
constexpr std::size_t kMinParallelElemsSdot3 = 262144;

// ========================================================================
// matvec_i4_sdot4_mt() — 多线程入口
// ========================================================================
void matvec_i4_sdot4_mt(const uint8_t *w, const float *x, float *y,
                        int out_dim, int in_dim, int group_size) {
    // 一次性量化激活
    const float scale_x = quantize_x_i8_v3(x, in_dim);

    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElemsSdot3 ||
        std::thread::hardware_concurrency() <= 1) {
        // 小矩阵单线程路径（结构与单线程入口相同）
        const int gpr = (in_dim + group_size - 1) / group_size;
        const int row_bytes = gpr * (kGroupHeader + group_size / 2);
        int o = 0;
        for (; o + 1 < out_dim; o += 2) {
            dot_2rows_i4_sdot4(w + static_cast<size_t>(o) * row_bytes,
                               w + static_cast<size_t>(o + 1) * row_bytes,
                               in_dim, group_size, scale_x, &y[o], &y[o + 1]);
        }
        if (o < out_dim) {
            y[o] = dot_row_i4_sdot4(w + static_cast<size_t>(o) * row_bytes,
                                    in_dim, group_size, scale_x);
        }
        return;
    }
    pool_sdot4().run(w, y, out_dim, in_dim, group_size, scale_x);
}

} // namespace

// 自注册：单线程 "sdot4" + 多线程 "sdot4_mt"
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot4, "sdot4");
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot4_mt, "sdot4_mt");

} // namespace tinyqwen

#endif // __aarch64__ && __ARM_FEATURE_DOTPROD
