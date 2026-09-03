// ============================================================================
// matvec_vq2_neon_mr_mt_wl.cpp — VQ2 matvec：neon_mr_mt + 索引 32 位字加载
// ============================================================================
// matvec_vq2_neon_mr_mt 的归因阶梯下一层。布局/线程池/粒度阈值全部不变，
// 只加一个技术：**索引按 32 位字批量加载**（配套 4 块展开）：
//
//   上一层内循环每块每行要 1 条 ldrb 读索引字节：
//     每块 9 load（1×x4 + 4×idx 字节 + 4×码本）对 4 FMA，
//     3 个 load 端口 → 下限 ~3 周期/块，load 发射是墙。
//   本层把块循环展开 4 倍，每行 4 个连续索引字节用一条 32 位加载读入：
//     每 4 块 24 load（4×x4 + 4×idx 字 + 16×码本）对 16 FMA，
//     → 6 load/块，load 下限降到 ~2 周期/块（-33%）。
//
// 索引区行内天然连续，memcpy 4 字节会被编译为单条 LDR（ARMv8 允许
// 非对齐）；行起点不保证 4 对齐（n_blocks 非 4 倍数时）也安全。
// 块数非 4 倍数的尾段退回逐字节路径。
//
// **尺寸门**：≥ kWordLoadMinIdxBytes(8MB) 走字加载，否则退回上一层的逐字节体
// （数值完全相同）。Qwen2.5-0.5B 各投影中仅 lm_head(34MB) 过门。原始依据是
// "中小形状（idx 驻留 L1/L2）被额外的移位/掩码 ALU 拖慢，896×896 倒退 2.3×"。
//
// **待定线索：8MB 可能定得过高**（2026-08-28 记录，未改默认值）
//   - 按本仓库自己的纪律分进程重测（`--impls ref,<单一实现>`，避免多个常驻
//     线程池同进程互扰——optimization_log 记载过这会造成 2× 失真），字加载在
//     896×896(196KB) 反而**快 1.45×**（0.016→0.011ms）、在 128×896(28KB) 与
//     逐字节体持平：**没能复现任何尺寸上的倒退**。
//   - 真实模型侧：Qwen3-0.6B 各层索引区仅 0.25–0.75 MB，全部被 8MB 门挡住；
//     而 decode 每 token 流过 110 MB 权重，这些"小"形状其实是 **DRAM 冷的**
//     （微基准把同一矩阵重复几千次，它常驻 L2——条件与真实推理相反）。
//   - 但端到端复测**不稳定**：一次成对测量 decode 11.10→10.41 ms/tok(1.064×)、
//     54-token 批量 prefill TTFT 472→357ms(1.33×)；持续压测后机器热噪声升到
//     ±3ms（同配置 10.59 vs 14.01），无法复现，且有一轮显示既有 Qwen3.5 VQ2
//     模型反而退化 1.07×。
//   → 按"优化后必须复测"的纪律默认值保持 8MB。推进这条线索需要冷机环境下对
//     两个 VQ2 模型各做多轮 ABA 成对测量。
//
// 数值语义与 neon_mr_mt 完全一致（同一查表 + fp32 累加，行内运算序不变：
// 每行仍按块序 b=0,1,2,... 累加，只是指令调度重排）。
//
// 仅 aarch64 构建注册；其他平台编译为空翻译单元。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_VQ2_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

constexpr int kBlockDim = 4;
constexpr int kCodebookEntries = 256;
constexpr int kCodebookBytes = kCodebookEntries * kBlockDim * 2; // 2048B
constexpr int kRows = 4;                                          // 多行并行度
constexpr int kUnroll = 4;                                        // 块展开度（=字宽）
constexpr int kNtUnroll = 16;                                     // nt 体展开度（=LDNP 宽）

// 行区间内循环的三种体。数值逐位一致，只在索引加载方式上不同。
enum class Body {
    kByte,  // 逐字节 ldrb
    kWl,    // 32 位字加载（1 条取 4 个索引）
    kNt,    // LDNP 流式（1 条取 16 个索引，带非临时提示）
};

// ========================================================================
// run_range_byte() — 小形状体：逐字节索引加载（与上一层 neon_mr_mt 同款）
// ========================================================================
inline void run_range_byte(const float cb[kCodebookEntries][kBlockDim],
                           const uint8_t *idx, int n_blocks,
                           const float *x, float *y, int begin, int end) {
    int o = begin;
    for (; o + kRows <= end; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), x4);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), x4);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), x4);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), x4);
        }
        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }
    for (; o < end; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int b = 0; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[row[b]]), x4);
        }
        y[o] = vaddvq_f32(acc);
    }
}

// ========================================================================
// run_range_wl() — 大形状体：4 行并行 × 4 块展开 × 32 位字索引加载
// ========================================================================
inline void run_range_wl(const float cb[kCodebookEntries][kBlockDim],
                         const uint8_t *idx, int n_blocks,
                         const float *x, float *y, int begin, int end) {
    int o = begin;
    for (; o + kRows <= end; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        int b = 0;
        // 主体：每次 4 块——每行一条 32 位加载取 4 个索引，省 3 条 ldrb
        for (; b + kUnroll <= n_blocks; b += kUnroll) {
            uint32_t w0, w1, w2, w3;
            std::memcpy(&w0, r0 + b, 4); // 单条 LDR（非对齐安全）
            std::memcpy(&w1, r1 + b, 4);
            std::memcpy(&w2, r2 + b, 4);
            std::memcpy(&w3, r3 + b, 4);

            const float32x4_t x0 = vld1q_f32(x + (b + 0) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[w0 & 0xFFu]), x0);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[w1 & 0xFFu]), x0);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[w2 & 0xFFu]), x0);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[w3 & 0xFFu]), x0);

            const float32x4_t x1 = vld1q_f32(x + (b + 1) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[(w0 >> 8) & 0xFFu]), x1);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[(w1 >> 8) & 0xFFu]), x1);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[(w2 >> 8) & 0xFFu]), x1);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[(w3 >> 8) & 0xFFu]), x1);

            const float32x4_t x2 = vld1q_f32(x + (b + 2) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[(w0 >> 16) & 0xFFu]), x2);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[(w1 >> 16) & 0xFFu]), x2);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[(w2 >> 16) & 0xFFu]), x2);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[(w3 >> 16) & 0xFFu]), x2);

            const float32x4_t x3 = vld1q_f32(x + (b + 3) * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[w0 >> 24]), x3);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[w1 >> 24]), x3);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[w2 >> 24]), x3);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[w3 >> 24]), x3);
        }
        // 块尾（不足 4 块）：逐字节路径（与上一层同款）
        for (; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), x4);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), x4);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), x4);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), x4);
        }

        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }

    // 尾行（区间内不足 4 行）：单行 + 同款字加载内循环
    for (; o < end; ++o) {
        const uint8_t *row = idx + static_cast<size_t>(o) * n_blocks;
        float32x4_t acc = vdupq_n_f32(0.0f);
        int b = 0;
        for (; b + kUnroll <= n_blocks; b += kUnroll) {
            uint32_t w;
            std::memcpy(&w, row + b, 4);
            float32x4_t xv = vld1q_f32(x + (b + 0) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[w & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 1) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[(w >> 8) & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 2) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[(w >> 16) & 0xFFu]), xv);
            xv = vld1q_f32(x + (b + 3) * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[w >> 24]), xv);
        }
        for (; b < n_blocks; ++b) {
            const float32x4_t x4 = vld1q_f32(x + b * kBlockDim);
            acc = vfmaq_f32(acc, vld1q_f32(cb[row[b]]), x4);
        }
        y[o] = vaddvq_f32(acc);
    }
}

// ========================================================================
// ldnp_u64x2() — 一条指令取 16 个索引（LDNP，GPR 对）
// ========================================================================
// **收益归因（已做隔离实验，勿再把功劳记错）**：
//   - **载入变宽是真收益**：一条取 16 个索引（逐字节体一条取 1 个、wl 体的
//     32 位 LDR 取 4 个）。每层 7 投影之和 0.313 → 0.205 ms = **1.53×**。
//   - **非临时提示这一项拿不出证据**。把本函数的 `ldnp` 换成同宽度的普通
//     `ldp` 做对照，lm_head(37MB，真流式) 上两组读数完全重叠：
//     ldp 1.853/1.753/1.570、ldnp 1.510/2.089/1.652——最优对最优 1.04×，
//     但 ldnp 有一轮比 ldp 全部读数都差。**效应在噪声之下。**
//   - 原假设是"码本必须常驻 L1，索引流用普通 load 会挤它"。该前提**也未经
//     验证**：码本 fp32 展开只有 4KB，落在 128KB 8 路 L1 里、每周期都在访问，
//     很可能压根没被挤出去过——那就正好解释了提示为什么不起作用。
//     要证实需要 L1D miss 的硬件计数器，本机没现成通路。
//   保留 `ldnp` 而非改回 `ldp`：两者同宽同对齐要求、实测不更差，且与 f32/f16
//   的 nt 级用同一条指令，命名一致。但**别指望非临时语义带来收益**。
//
// 用 GPR 对而非 q 寄存器对：索引要参与地址计算，留在通用寄存器里才不用
// 付向量→GPR 搬运的代价（那在 aarch64 上是每字节一条 umov）。
// clang 的 __builtin_nontemporal_load 对此静默丢提示，故内联汇编显式发射。
inline void ldnp_u64x2(const uint8_t *p, std::uint64_t &lo, std::uint64_t &hi) {
    __asm__("ldnp %[lo], %[hi], [%[p]]"
            : [lo] "=&r"(lo), [hi] "=&r"(hi)
            : [p] "r"(p));
}

// ========================================================================
// run_range_nt() — 流式体：索引走 LDNP（16 块展开），x 与码本走普通加载
// ========================================================================
// 数值与 byte/wl 两体**逐位一致**：同一码本查表、同一 x、每行仍按块序
// b=0,1,2,... 累加，只是索引加载指令换宽、加了非临时提示。
// 字节序与 wl 体同款：LDNP 按小端装入，第 j 个字节 = (w >> 8j) & 0xFF。
inline void run_range_nt(const float cb[kCodebookEntries][kBlockDim],
                         const uint8_t *idx, int n_blocks,
                         const float *x, float *y, int begin, int end) {
    int o = begin;
    for (; o + kRows <= end; o += kRows) {
        const uint8_t *r0 = idx + static_cast<size_t>(o + 0) * n_blocks;
        const uint8_t *r1 = idx + static_cast<size_t>(o + 1) * n_blocks;
        const uint8_t *r2 = idx + static_cast<size_t>(o + 2) * n_blocks;
        const uint8_t *r3 = idx + static_cast<size_t>(o + 3) * n_blocks;

        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = vdupq_n_f32(0.0f);
        float32x4_t a2 = vdupq_n_f32(0.0f);
        float32x4_t a3 = vdupq_n_f32(0.0f);

        int b = 0;
        // 主体：每次 16 块——每行一条 LDNP 取 16 个索引
        for (; b + kNtUnroll <= n_blocks; b += kNtUnroll) {
            std::uint64_t l0, h0, l1, h1, l2, h2, l3, h3;
            ldnp_u64x2(r0 + b, l0, h0);
            ldnp_u64x2(r1 + b, l1, h1);
            ldnp_u64x2(r2 + b, l2, h2);
            ldnp_u64x2(r3 + b, l3, h3);
// 一个块：4 行各查一次码本 + 一次 FMA，共享同一个 x 向量
#define TQ_VQ2_NT_BLK(K, W0, W1, W2, W3, SH)                                          \
    {                                                                                  \
        const float32x4_t xk = vld1q_f32(x + (b + (K)) * kBlockDim);                    \
        a0 = vfmaq_f32(a0, vld1q_f32(cb[((W0) >> (SH)) & 0xFFu]), xk);                  \
        a1 = vfmaq_f32(a1, vld1q_f32(cb[((W1) >> (SH)) & 0xFFu]), xk);                  \
        a2 = vfmaq_f32(a2, vld1q_f32(cb[((W2) >> (SH)) & 0xFFu]), xk);                  \
        a3 = vfmaq_f32(a3, vld1q_f32(cb[((W3) >> (SH)) & 0xFFu]), xk);                  \
    }
            TQ_VQ2_NT_BLK(0, l0, l1, l2, l3, 0)
            TQ_VQ2_NT_BLK(1, l0, l1, l2, l3, 8)
            TQ_VQ2_NT_BLK(2, l0, l1, l2, l3, 16)
            TQ_VQ2_NT_BLK(3, l0, l1, l2, l3, 24)
            TQ_VQ2_NT_BLK(4, l0, l1, l2, l3, 32)
            TQ_VQ2_NT_BLK(5, l0, l1, l2, l3, 40)
            TQ_VQ2_NT_BLK(6, l0, l1, l2, l3, 48)
            TQ_VQ2_NT_BLK(7, l0, l1, l2, l3, 56)
            TQ_VQ2_NT_BLK(8, h0, h1, h2, h3, 0)
            TQ_VQ2_NT_BLK(9, h0, h1, h2, h3, 8)
            TQ_VQ2_NT_BLK(10, h0, h1, h2, h3, 16)
            TQ_VQ2_NT_BLK(11, h0, h1, h2, h3, 24)
            TQ_VQ2_NT_BLK(12, h0, h1, h2, h3, 32)
            TQ_VQ2_NT_BLK(13, h0, h1, h2, h3, 40)
            TQ_VQ2_NT_BLK(14, h0, h1, h2, h3, 48)
            TQ_VQ2_NT_BLK(15, h0, h1, h2, h3, 56)
#undef TQ_VQ2_NT_BLK
        }
        // 块尾：先用 wl 的 4 块字体，再逐字节
        for (; b + kUnroll <= n_blocks; b += kUnroll) {
            std::uint32_t w0, w1, w2, w3;
            std::memcpy(&w0, r0 + b, 4);
            std::memcpy(&w1, r1 + b, 4);
            std::memcpy(&w2, r2 + b, 4);
            std::memcpy(&w3, r3 + b, 4);
            for (int k = 0; k < kUnroll; ++k) {
                const unsigned sh = static_cast<unsigned>(k) * 8u;
                const float32x4_t xk = vld1q_f32(x + (b + k) * kBlockDim);
                a0 = vfmaq_f32(a0, vld1q_f32(cb[(w0 >> sh) & 0xFFu]), xk);
                a1 = vfmaq_f32(a1, vld1q_f32(cb[(w1 >> sh) & 0xFFu]), xk);
                a2 = vfmaq_f32(a2, vld1q_f32(cb[(w2 >> sh) & 0xFFu]), xk);
                a3 = vfmaq_f32(a3, vld1q_f32(cb[(w3 >> sh) & 0xFFu]), xk);
            }
        }
        for (; b < n_blocks; ++b) {
            const float32x4_t xk = vld1q_f32(x + b * kBlockDim);
            a0 = vfmaq_f32(a0, vld1q_f32(cb[r0[b]]), xk);
            a1 = vfmaq_f32(a1, vld1q_f32(cb[r1[b]]), xk);
            a2 = vfmaq_f32(a2, vld1q_f32(cb[r2[b]]), xk);
            a3 = vfmaq_f32(a3, vld1q_f32(cb[r3[b]]), xk);
        }

        y[o + 0] = vaddvq_f32(a0);
        y[o + 1] = vaddvq_f32(a1);
        y[o + 2] = vaddvq_f32(a2);
        y[o + 3] = vaddvq_f32(a3);
    }
    // 尾行（不足 4 行）：交给逐字节体，最多 3 行，与主体逐位一致
    if (o < end) run_range_byte(cb, idx, n_blocks, x, y, o, end);
}

// 码本 [K, d] fp16 → fp32（与上一层同款）
inline void load_codebook(const uint8_t *w, float cb[kCodebookEntries][kBlockDim]) {
    for (int k = 0; k < kCodebookEntries; ++k) {
        for (int j = 0; j < kBlockDim; ++j) {
            uint16_t h;
            std::memcpy(&h, w + (k * kBlockDim + j) * 2, 2);
            cb[k][j] = half_to_float(h);
        }
    }
}

inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
    int spins = 0;
    while (a.load(std::memory_order_acquire) != target) {
        if (++spins <= 256) {
            __builtin_arm_yield();
        } else {
            std::this_thread::yield();
        }
    }
}

inline int default_parallelism_vq2_wl() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolVQ2Wl — 常驻行切分线程池（结构同 RowPoolVQ2，内循环换成字加载版）
// ========================================================================
struct RowPoolVQ2Wl {
    float cb[kCodebookEntries][kBlockDim];
    const uint8_t *idx_base = nullptr;
    const float *x = nullptr;
    float *y = nullptr;
    int out_dim = 0;
    int n_blocks = 0;
    Body body = Body::kByte;   // 内循环体：byte / wl / nt

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolVQ2Wl() {
        const int p = default_parallelism_vq2_wl();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx_ = 1; idx_ < p; ++idx_) {
            workers.emplace_back([this, idx_] { worker_main(idx_); });
        }
    }

    ~RowPoolVQ2Wl() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_chunk(idx);
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    void do_chunk(int idx) const {
        const int p = static_cast<int>(workers.size()) + 1;
        const int base = out_dim / p;
        const int rem = out_dim % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        if (body == Body::kNt) run_range_nt(cb, idx_base, n_blocks, x, y, begin, end);
        else if (body == Body::kWl) run_range_wl(cb, idx_base, n_blocks, x, y, begin, end);
        else run_range_byte(cb, idx_base, n_blocks, x, y, begin, end);
    }

    void run(const uint8_t *w, const float *x_, float *y_, int out_dim_, int in_dim_,
             Body body_) {
        load_codebook(w, cb);
        idx_base = w + kCodebookBytes;
        x = x_;
        y = y_;
        out_dim = out_dim_;
        n_blocks = in_dim_ / kBlockDim;
        body = body_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until(done_gen, expected_done);
    }
};

RowPoolVQ2Wl &pool_vq2_wl() {
    static RowPoolVQ2Wl p;
    return p;
}

// 粒度阈值（按元素数，与上一层同款）
constexpr std::size_t kMinParallelElems = 262144;

// 字加载体的尺寸门：索引区字节数 ≥ 64KB 才用字加载体。
// 只是给退化小形状留兜底（实测那里两条体等速），不是性能分流——依据见文件头。
constexpr std::size_t kWordLoadMinIdxBytes = 8u << 20;

} // namespace

// ========================================================================
// 共用实现：按 body 跑（小形状/单核直接串行，否则交常驻线程池）
// ========================================================================
namespace {
void run_vq2_wl_family(const uint8_t *w, const float *x, float *y,
                       int out_dim, int in_dim, Body body) {
    const int n_blocks = in_dim / kBlockDim;
    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElems ||
        std::thread::hardware_concurrency() <= 1) {
        float cb[kCodebookEntries][kBlockDim];
        load_codebook(w, cb);
        const uint8_t *idx = w + kCodebookBytes;
        if (body == Body::kNt) run_range_nt(cb, idx, n_blocks, x, y, 0, out_dim);
        else if (body == Body::kWl) run_range_wl(cb, idx, n_blocks, x, y, 0, out_dim);
        else run_range_byte(cb, idx, n_blocks, x, y, 0, out_dim);
        return;
    }
    pool_vq2_wl().run(w, x, y, out_dim, in_dim, body);
}

// 尺寸门：索引区字节数（= out_dim × n_blocks）≥ 8MB 才值得字加载
inline Body wl_body_for(int out_dim, int n_blocks) {
    return static_cast<std::size_t>(out_dim) * n_blocks >= kWordLoadMinIdxBytes
                   ? Body::kWl : Body::kByte;
}
} // namespace

// ========================================================================
// matvec_vq2_neon_mr_mt_wl() — VQ2 matvec：4 行并行 + 多线程 + 字加载入口
// ========================================================================
void matvec_vq2_neon_mr_mt_wl(const uint8_t *w, const float *x, float *y,
                              int out_dim, int in_dim) {
    run_vq2_wl_family(w, x, y, out_dim, in_dim,
                      wl_body_for(out_dim, in_dim / kBlockDim));
}

// 自注册进 dispatch：matvec_vq2 的 "neon_mr_mt_wl" 实现
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon_mr_mt_wl, "neon_mr_mt_wl");

// ========================================================================
// matvec_vq2_neon_mr_mt_wl_nt() — 上一级 + 索引流 LDNP（非临时加载）
// ========================================================================
// LDNP 用 GPR 对，要求行起点 8 字节对齐。行起点 = idx_base + row × n_blocks，
// 故需 idx_base 8 对齐且 n_blocks 是 8 的倍数（真实模型都满足：数据区 64 对齐、
// 码本 2048B、n_blocks = in_dim/4 且 in_dim ≥ 32 时为 8 的倍数）。
// 不满足就退回字加载/逐字节体——与 f32 的 nt 级同款守卫，数值逐位一致。
void matvec_vq2_neon_mr_mt_wl_nt(const uint8_t *w, const float *x, float *y,
                                 int out_dim, int in_dim) {
    const int n_blocks = in_dim / kBlockDim;
    const bool aligned = (reinterpret_cast<std::uintptr_t>(w + kCodebookBytes) % 8u) == 0 &&
                         (n_blocks % 8) == 0;
    const Body body = (aligned && n_blocks >= kNtUnroll) ? Body::kNt
                                                         : wl_body_for(out_dim, n_blocks);
    run_vq2_wl_family(w, x, y, out_dim, in_dim, body);
}

// 自注册进 dispatch：matvec_vq2 的 "neon_mr_mt_wl_nt" 实现
TINYQWEN_MATVEC_VQ2_VARIANT(matvec_vq2_neon_mr_mt_wl_nt, "neon_mr_mt_wl_nt");

} // namespace tinyqwen

#endif // __aarch64__
