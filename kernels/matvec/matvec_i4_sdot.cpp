// INT4 weight-only matvec：y = W @ x —— W4A8 SDOT 版（armv8.2 dotprod）。
//
// 布局与 ref 一致：每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(G/2 B)]，
// 反量化语义 w = (q - zero) * scale，q ∈ [0,15]。
//
// 为什么快（对比 matvec_i4_neon 的 ~3.5 指令/字节展宽链）：
//   不把权重转 fp32，而是走整数点积 SDOT（1 条指令做 16 个 int8×int8 乘累加）。
//   - 权重：q_s = q - 8 ∈ [-8,7]（有符号 int8），unpack 仅 vand/vshr/vzip/vsub；
//   - 激活：x 每次 matvec 对称量化成 int8（scale_x = max|x|/127），常驻 scratch；
//   - 非对称 zero 修正：y[o] = Σ_g A_g·DOT_g − A_g·(zero_w[g]−8)·XQSUM_g，
//     其中 DOT_g = Σ q_s·x_q（SDOT），XQSUM_g = Σ x_q（x_q 前缀和，预计算），
//     A_g = scale_w[g]·scale_x。
//   指令/字节降到 ~0.9，单核 dequant 上限显著抬升（目标逼近带宽墙）。
//
// 代价：激活 int8 量化引入量化误差（业界 W4A8 标准做法，端到端 token 门禁把关）。
// group_size 必须是 16 的倍数（SDOT 每次 16 元素）。

#include "dispatch.h"
#include "ref_ops.h"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

constexpr int kGroupHeader = 4;
constexpr int kMaxInDim = 8192;

// 每次 matvec 调用复用的 scratch：int8 激活 + int32 前缀和（串行调用，静态安全）。
int8_t  g_xq[kMaxInDim];
int32_t g_xq_prefix[kMaxInDim + 1];

// 对称 int8 量化 x，并填前缀和。返回 scale_x。
float quantize_x_i8(const float *x, int in_dim) {
    const int n = in_dim < kMaxInDim ? in_dim : kMaxInDim;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale_x;
    g_xq_prefix[0] = 0;
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv;
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);
        g_xq[i] = static_cast<int8_t>(q);
        g_xq_prefix[i + 1] = g_xq_prefix[i] + q;
    }
    return scale_x;
}

// 一行点积（W4A8 SDOT）。
float dot_row_i4_sdot(const uint8_t *row, int in_dim, int group_size,
                      float scale_x) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total_bytes = kGroupHeader + group_size / 2;
    float acc = 0.0f;
    int col = 0;

    const uint8x16_t v8 = vdupq_n_u8(8);
    const uint8x8_t mask_lo = vdup_n_u8(0x0F);

    for (int g = 0; g < groups_per_row; ++g) {
        const uint8_t *gp = row + static_cast<size_t>(g) * group_total_bytes;
        uint16_t scale_h, zero_h;
        std::memcpy(&scale_h, gp, 2);
        std::memcpy(&zero_h, gp + 2, 2);
        const float scale_w = half_to_float(scale_h);
        const float zero_w = half_to_float(zero_h);
        const float A = scale_w * scale_x;              // Σ q_s·x_q 的系数
        const float C = A * (zero_w - 8.0f);            // XQSUM 的系数（zero 修正）

        const uint8_t *packed = gp + kGroupHeader;
        const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

        int32x4_t acc_dot = vdupq_n_s32(0);
        int i = 0;
        const int n16 = group_elems & ~15;
        for (; i < n16; i += 16) {
            // 8 bytes packed = 16 nibble = 16 weights
            uint8x8_t raw = vld1_u8(packed + i / 2);
            uint8x8_t lo = vand_u8(raw, mask_lo);       // w0,w2,...,w14
            uint8x8_t hi = vshr_n_u8(raw, 4);           // w1,w3,...,w15
            uint8x8x2_t z = vzip_u8(lo, hi);            // val[0]=w0..w7, val[1]=w8..w15
            uint8x16_t q_u = vcombine_u8(z.val[0], z.val[1]);
            int8x16_t q_s = vreinterpretq_s8_u8(vsubq_u8(q_u, v8)); // q-8 → 有符号
            int8x16_t xq = vld1q_s8(g_xq + col + i);
            acc_dot = vdotq_s32(acc_dot, q_s, xq);   // SDOT：16 个 int8×int8 → 4 路 int32
        }
        int dot_g = vaddvq_s32(acc_dot);

        // 标量尾部（group_elems 非 16 倍数时）
        for (; i < group_elems; ++i) {
            const int byte_idx = i / 2;
            const int q = (i % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                       : ((packed[byte_idx] >> 4) & 0x0F);
            dot_g += (q - 8) * static_cast<int>(g_xq[col + i]);
        }

        const int xqsum_g = g_xq_prefix[col + group_elems] - g_xq_prefix[col];
        acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum_g);
        col += group_size;
    }
    return acc;
}

void matvec_i4_sdot(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);

    const float scale_x = quantize_x_i8(x, in_dim);

    for (int o = 0; o < out_dim; ++o) {
        y[o] = dot_row_i4_sdot(w + static_cast<size_t>(o) * row_bytes,
                               in_dim, group_size, scale_x);
    }
}

// ---- 多线程版 sdot_mt：RowPool 行切分，复用单线程的 dot_row_i4_sdot ----
// 全局 scratch（g_xq/g_xq_prefix）在 quantize_x_i8 一次性填充后对所有线程只读，
// 线程安全；quantize_x_i8 在主线程并行段之前调用一次。
// （本段仍在上方匿名 namespace 内，与单线程实现同域。）

inline void spin_until_sdot(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
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

inline int default_parallelism_sdot() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

struct RowPoolSdot {
    const uint8_t *w = nullptr;
    float *y = nullptr;
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 64;
    float scale_x = 1.0f;

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    RowPoolSdot() {
        const int p = default_parallelism_sdot();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ~RowPoolSdot() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t: workers) t.join();
    }

    void worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_sdot(job_gen, next_job);
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
        const int groups_per_row = (in_dim + group_size - 1) / group_size;
        const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);
        for (int o = begin; o < end; ++o) {
            y[o] = dot_row_i4_sdot(w + static_cast<size_t>(o) * row_bytes,
                                   in_dim, group_size, scale_x);
        }
    }

    void run(const uint8_t *w_, float *y_, int out_dim_, int in_dim_,
             int group_size_, float scale_x_) {
        w = w_;
        y = y_;
        out_dim = out_dim_;
        in_dim = in_dim_;
        group_size = group_size_;
        scale_x = scale_x_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until_sdot(done_gen, expected_done);
    }
};

RowPoolSdot &pool_sdot() {
    static RowPoolSdot p;
    return p;
}

constexpr std::size_t kMinParallelElemsSdot = 262144;

void matvec_i4_sdot_mt(const uint8_t *w, const float *x, float *y,
                       int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);
    // 一次性量化激活（填全局 scratch，之后对所有线程只读）。
    const float scale_x = quantize_x_i8(x, in_dim);

    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElemsSdot ||
        std::thread::hardware_concurrency() <= 1) {
        for (int o = 0; o < out_dim; ++o) {
            y[o] = dot_row_i4_sdot(w + static_cast<size_t>(o) * row_bytes,
                                   in_dim, group_size, scale_x);
        }
        return;
    }
    pool_sdot().run(w, y, out_dim, in_dim, group_size, scale_x);
}

} // namespace

TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot, "sdot");
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot_mt, "sdot_mt");

} // namespace tinyqwen

#endif // __aarch64__ && __ARM_FEATURE_DOTPROD
