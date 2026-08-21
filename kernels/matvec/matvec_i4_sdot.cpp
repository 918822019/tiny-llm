// ============================================================================
// matvec_i4_sdot.cpp — INT4 weight-only matvec：y = W @ x
//                  W4A8 SDOT 版（armv8.2 dotprod）+ 多线程版
// ============================================================================
// 本文件实现两种变体：单线程 "sdot" 和多线程 "sdot_mt"。
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
//
// 多线程版 sdot_mt：RowPool 行切分，复用单线程的 dot_row_i4_sdot。
// 全局 scratch（g_xq/g_xq_prefix）在 quantize_x_i8 一次性填充后对所有线程只读，
// 线程安全；quantize_x_i8 在主线程并行段之前调用一次。
//
// 条件编译：需要 __aarch64__ && __ARM_FEATURE_DOTPROD（armv8.2-a dotprod 扩展）。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)

#include <arm_neon.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace tinyqwen {
namespace {

constexpr int kGroupHeader = 4;    // 每组头部字节数
constexpr int kMaxInDim = 8192;    // 支持的最大输入维度

// ========================================================================
// 全局 scratch 缓冲区（串行调用，静态安全）
// ========================================================================
// g_xq：int8 量化后的激活向量
// g_xq_prefix：x_q 的前缀和数组（用于快速计算组内 x_q 之和）
int8_t  g_xq[kMaxInDim];           // 对称 int8 量化的激活
int32_t g_xq_prefix[kMaxInDim + 1]; // 前缀和：g_xq_prefix[i] = Σ_{j=0}^{i-1} g_xq[j]

// ========================================================================
// quantize_x_i8() — 对称 int8 量化 x，并填前缀和
// ========================================================================
// 功能：将 fp32 输入向量 x 对称量化为 int8，同时计算前缀和。
// 参数：x — fp32 输入向量, in_dim — 向量长度
// 返回值：scale_x（量化缩放因子 = max|x| / 127）
// 说明：对称量化意味着 zero_point = 0，只需一个 scale。
//       前缀和使得任意区间 [a,b) 的 x_q 之和可以 O(1) 查询。
float quantize_x_i8(const float *x, int in_dim) {
    const int n = in_dim < kMaxInDim ? in_dim : kMaxInDim; // 截断到最大维度
    // 第一步：找绝对值最大值（amax）
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i]; // 取绝对值
        if (a > amax) amax = a;
    }
    // 计算 scale_x：将 [-amax, amax] 映射到 [-127, 127]
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = 1.0f / scale_x; // 逆 scale（乘法比除法快）
    // 初始化前缀和
    g_xq_prefix[0] = 0;
    // 第二步：逐元素量化 + 填前缀和
    for (int i = 0; i < n; ++i) {
        float v = x[i] * inv; // 缩放到 [-127, 127] 范围
        // clamp 到 [-127, 127]
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
        // 四舍五入到最近的整数
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);
        g_xq[i] = static_cast<int8_t>(q); // 存入 int8 缓冲区
        g_xq_prefix[i + 1] = g_xq_prefix[i] + q; // 更新前缀和
    }
    return scale_x;
}

// ========================================================================
// dot_row_i4_sdot() — 一行点积（W4A8 SDOT）
// ========================================================================
// 功能：用 SDOT 指令加速的单行 INT4 点积
// 参数：
//   row       — 该行的 packed 权重起始地址
//   in_dim    — 输入维度
//   group_size — 量化分组大小
//   scale_x   — 激活的量化缩放因子
// 返回值：该行的输出值（fp32）
// 算法：对每个 group，用 SDOT 算整数点积 DOT_g = Σ q_s·x_q，
//       再用公式 y += A·DOT - C·XQSUM 还原为浮点结果。
float dot_row_i4_sdot(const uint8_t *row, int in_dim, int group_size,
                      float scale_x) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total_bytes = kGroupHeader + group_size / 2;
    float acc = 0.0f; // 浮点累加器
    int col = 0;      // 当前列位置

    const uint8x16_t v8 = vdupq_n_u8(8);       // 常量 8（用于 q-8 偏移）
    const uint8x8_t mask_lo = vdup_n_u8(0x0F); // 低 nibble 掩码（64 位版本）

    for (int g = 0; g < groups_per_row; ++g) {
        const uint8_t *gp = row + static_cast<size_t>(g) * group_total_bytes;
        // 读取该组的 scale 和 zero_point（fp16 → fp32）
        uint16_t scale_h, zero_h;
        std::memcpy(&scale_h, gp, 2);
        std::memcpy(&zero_h, gp + 2, 2);
        const float scale_w = half_to_float(scale_h);
        const float zero_w = half_to_float(zero_h);
        // A = scale_w * scale_x：整数点积 DOT_g 的缩放系数
        const float A = scale_w * scale_x;
        // C = A * (zero_w - 8)：zero 修正项的系数
        // 推导：真实值 = (q - zero) * scale_w * x_real
        //             = (q_s + 8 - zero) * scale_w * scale_x * x_q
        //             = q_s * x_q * A + (8 - zero) * A * x_q
        //             = q_s * x_q * A - (zero - 8) * A * x_q
        // 所以 y += A * DOT_g - C * XQSUM_g
        const float C = A * (zero_w - 8.0f);

        const uint8_t *packed = gp + kGroupHeader; // packed 数据起点
        const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

        // SDOT 累加器：4 路 int32（SDOT 每次产出 4 个 int32 部分和）
        int32x4_t acc_dot = vdupq_n_s32(0);
        int i = 0;
        const int n16 = group_elems & ~15; // 主循环边界：16 的倍数
        for (; i < n16; i += 16) {
            // 加载 8 字节 packed = 16 个 nibble = 16 个权重
            uint8x8_t raw = vld1_u8(packed + i / 2);
            // 拆分高低 nibble
            uint8x8_t lo = vand_u8(raw, mask_lo);       // w0,w2,...,w14（偶数下标）
            uint8x8_t hi = vshr_n_u8(raw, 4);           // w1,w3,...,w15（奇数下标）
            // 交错重排为顺序 [w0,w1,w2,...,w15]
            uint8x8x2_t z = vzip_u8(lo, hi);
            // 合并为 128 位向量
            uint8x16_t q_u = vcombine_u8(z.val[0], z.val[1]);
            // q_s = q_u - 8：将无符号 [0,15] 转为有符号 [-8,7]
            int8x16_t q_s = vreinterpretq_s8_u8(vsubq_u8(q_u, v8));
            // 加载对应的 int8 量化激活
            int8x16_t xq = vld1q_s8(g_xq + col + i);
            // vdotq_s32：SDOT 指令，16 个 int8×int8 乘累加到 4 路 int32
            // acc_dot[lane] += Σ_{k=0}^{3} q_s[lane*4+k] * xq[lane*4+k]
            acc_dot = vdotq_s32(acc_dot, q_s, xq);
        }
        // 横向归约：4 路 int32 加成一个标量
        int dot_g = vaddvq_s32(acc_dot);

        // 标量尾部（group_elems 非 16 倍数时）
        for (; i < group_elems; ++i) {
            const int byte_idx = i / 2;
            const int q = (i % 2 == 0) ? (packed[byte_idx] & 0x0F)
                                       : ((packed[byte_idx] >> 4) & 0x0F);
            dot_g += (q - 8) * static_cast<int>(g_xq[col + i]);
        }

        // 用前缀和 O(1) 查询该组的 x_q 之和
        const int xqsum_g = g_xq_prefix[col + group_elems] - g_xq_prefix[col];
        // 还原浮点结果：y += A * DOT_g - C * XQSUM_g
        acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum_g);
        col += group_size;
    }
    return acc;
}

// ========================================================================
// matvec_i4_sdot() — 单线程 SDOT matvec 入口
// ========================================================================
void matvec_i4_sdot(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);

    // 一次性量化激活（填全局 scratch，之后所有行共用）
    const float scale_x = quantize_x_i8(x, in_dim);

    // 逐行计算
    for (int o = 0; o < out_dim; ++o) {
        y[o] = dot_row_i4_sdot(w + static_cast<size_t>(o) * row_bytes,
                               in_dim, group_size, scale_x);
    }
}

// ========================================================================
// 多线程版 sdot_mt：RowPool 行切分
// ========================================================================
// 全局 scratch（g_xq/g_xq_prefix）在 quantize_x_i8 一次性填充后对所有线程只读，
// 线程安全。

// 自旋等待
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

// 默认并行度
inline int default_parallelism_sdot() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ========================================================================
// RowPoolSdot — 常驻行切分线程池（SDOT 版）
// ========================================================================
struct RowPoolSdot {
    const uint8_t *w = nullptr; // INT4 packed 权重
    float *y = nullptr;         // 输出向量
    int out_dim = 0;
    int in_dim = 0;
    int group_size = 64;
    float scale_x = 1.0f;       // 激活量化缩放因子（量化后对所有线程只读）

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

    // 第 idx 块行区间上的逐行 SDOT 点积
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

    // fork-join 入口
    void run(const uint8_t *w_, float *y_, int out_dim_, int in_dim_,
             int group_size_, float scale_x_) {
        w = w_; y = y_;
        out_dim = out_dim_; in_dim = in_dim_;
        group_size = group_size_; scale_x = scale_x_;
        job_gen.store(++job_counter, std::memory_order_release);
        do_chunk(0);
        expected_done += workers.size();
        spin_until_sdot(done_gen, expected_done);
    }
};

// Meyers singleton
RowPoolSdot &pool_sdot() {
    static RowPoolSdot p;
    return p;
}

// 粒度阈值
constexpr std::size_t kMinParallelElemsSdot = 262144;

// ========================================================================
// matvec_i4_sdot_mt() — 多线程 SDOT matvec 入口
// ========================================================================
void matvec_i4_sdot_mt(const uint8_t *w, const float *x, float *y,
                       int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int row_bytes = groups_per_row * (kGroupHeader + group_size / 2);
    // 一次性量化激活（填全局 scratch，之后对所有线程只读）
    const float scale_x = quantize_x_i8(x, in_dim);

    if (static_cast<std::size_t>(out_dim) * in_dim < kMinParallelElemsSdot ||
        std::thread::hardware_concurrency() <= 1) {
        // 小矩阵走单线程
        for (int o = 0; o < out_dim; ++o) {
            y[o] = dot_row_i4_sdot(w + static_cast<size_t>(o) * row_bytes,
                                   in_dim, group_size, scale_x);
        }
        return;
    }
    pool_sdot().run(w, y, out_dim, in_dim, group_size, scale_x);
}

} // namespace

// 自注册：单线程 "sdot" + 多线程 "sdot_mt"
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot, "sdot");
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_sdot_mt, "sdot_mt");

} // namespace tinyqwen

#endif // __aarch64__ && __ARM_FEATURE_DOTPROD
