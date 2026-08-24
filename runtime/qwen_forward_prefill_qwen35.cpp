// ============================================================================
// qwen_forward_prefill_qwen35.cpp — Qwen3.5 混合架构的批量 prefill（GEMM 路径）
// ============================================================================
// 背景：Qwen3.5 的 prefill 此前逐 token 回退（GDN 状态需顺序更新），每个
// prompt token 都把全部权重读一遍——4B 上 33-token prompt 的 TTFT ≈ 33 ×
// decode 单 token 时间。本文件实现批量路径：**权重每层只读一遍**（反量化到
// fp32 后走 BLAS GEMM，Apple Silicon 上是 Accelerate/AMX），把线性投影摊薄
// 到全部 token 上；只有真正跨 token 耦合的部分保留顺序扫描：
//
//   - GDN 层：conv1d 状态与递归状态矩阵 S 必须按 token 顺序更新
//     → 投影（in_proj_qkv/z/b/a、out_proj）批量 GEMM，
//       conv1d + l2norm + delta rule + 门控 norm 逐 token 扫描（复用
//       单 token 算子，数值契约与 forward_token 完全一致）。
//   - Full attention 层：因果性使每个 token 只能看自己的前缀
//     → q/k/v/o 投影批量 GEMM，QK-norm/partial RoPE/KV 追加/attention
//       逐 token 扫描（复用单 token 算子）。
//   - FFN：无跨 token 耦合，整层批量（norm/gate/up/swiglu/down）。
//
// 数值说明：GEMM 的 fp32 累加顺序与逐 token matvec 不同，结果有 ~1e-6 量级
// 差异（与 HF 的批量前向同属一类），贪心输出在实测中与逐 token 路径一致；
// 对照验证走 tools/align_fake_qwen35_model.py --batch 与 4B 双路 A/B。
//
// GEMM 后端：Apple Silicon 用 Accelerate（cblas_sgemm，自动 AMX/多核）。
// 其他平台（如 Android）当前返回"不可用"，调用方回退逐 token 路径。
//
// 权重支持：
//   - kI4：按组反量化到 fp32 scratch（NEON），再 GEMM；
//   - kF32：直接 GEMM（零转换）；
//   - kF16：fp16→fp32 转换到 scratch 后 GEMM。
// ============================================================================

#include "qwen_model.h"

#include <atomic>     // 反量化线程池的原子行计数 / job 代际
#include <cmath>      // std::sqrt, std::exp, std::log1p
#include <cstdint>
#include <cstdio>     // fprintf
#include <cstring>    // memcpy
#include <numeric>    // std::iota
#include <thread>     // 反量化线程池
#include <vector>

#include "dispatch.h"  // matvec_f32, argmax
#include "ref_ops.h"   // half_to_float

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#define TINYQWEN_HAS_CBLAS 1
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace tinyqwen {
namespace {

// ========================================================================
// 标量门控函数（与 qwen_forward_token.cpp 同定义，保证数值一致）
// ========================================================================
inline float sigmoidf32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

inline float softplusf32(float x) {
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

// top-k（与 forward_token 同实现；匿名命名空间各自独立不冲突）
void top_k_logits_pf(const float *logits, int vocab, int k, TopKResult *out) {
    k = std::min(k, vocab);
    std::vector<int> idx(vocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    out->indices.resize(k);
    out->values.resize(k);
    for (int i = 0; i < k; ++i) {
        out->indices[i] = idx[i];
        out->values[i] = logits[idx[i]];
    }
}

// ========================================================================
// fp16 位模式 → fp32（优先硬件转换）
// ========================================================================
// 与 sdot4 相同的守卫教训：clang +fp16 定义的是
// __ARM_FEATURE_FP16_SCALAR_ARITHMETIC，没有 __ARM_FEATURE_FP16。
inline float half_bits_to_float_pf(uint16_t h) {
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC)
    _Float16 hf;
    std::memcpy(&hf, &h, sizeof(hf));
    return static_cast<float>(hf);
#else
    return half_to_float(h);
#endif
}

// ========================================================================
// gemm_wx: Y[N,M] = X[N,K] @ W^T（token 主序）
// ========================================================================
// x — [N,K] token 主序（第 t 个 token 的 K 维向量在 t*K）
// w — [M,K] 行主序权重（第 m 个输出神经元在 m*K）
// y — [N,M] token 主序（第 t 个 token 的 M 维输出在 t*M）
// 返回 false 表示平台无 GEMM 后端。
bool gemm_wx(const float *w, const float *x, float *y, int M, int K, int N) {
#if defined(TINYQWEN_HAS_CBLAS)
    // 行主序形式：Y^T[N,M] 不需要——直接算 y_rm[N,M] = x_rm[N,K] @ (w_rm[M,K])^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, M, K,
                1.0f, x, K, w, K, 0.0f, y, M);
    return true;
#else
    (void)w; (void)x; (void)y; (void)M; (void)K; (void)N;
    return false;
#endif
}

} // namespace

namespace {

// ========================================================================
// 并行反量化：行区间函数 + 常驻工作窃取线程池
// ========================================================================
// 反量化是批量 prefill 的固定开销（每层每个权重矩阵一遍），行之间天然
// 独立——串行时是单线程瓶颈（4B 上 ~0.7s），按行切到常驻线程池后可打满
// DRAM 墙。结构与 matvec 的 RowPool 同款（原子行计数 + 自旋等待）。

// 低于此元素数走串行（池同步开销不划算；gdn_in_b/a 这类小矩阵）
constexpr std::size_t kMinParallelElemsDq = 262144;

// 自旋等待（与 matvec 池相同策略：先 YIELD 再让出）
inline void spin_until_dq(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
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

// 并行度：与 matvec 池同约定（TINYQWEN_MT_THREADS 可覆盖，上限 16）
inline int default_parallelism_dq() {
    if (const char *env = std::getenv("TINYQWEN_MT_THREADS")) {
        const long v = std::strtol(env, nullptr, 10);
        if (v >= 1) return static_cast<int>(v > 16 ? 16 : v);
    }
    int p = static_cast<int>(std::thread::hardware_concurrency());
    if (p <= 1) p = 2;
    return p > 16 ? 16 : p;
}

// ------------------------------------------------------------------------
// dequant_i4_rows: 反量化 [M,K] 权重的第 [row_begin, row_end) 行
// ------------------------------------------------------------------------
// 磁盘布局（每行每组）：[scale_fp16(2B) | zero_fp16(2B) | packed(K/2 B)]
// 反量化语义（与导出器/HF 一致）：value = (uint4 - zero) * scale
void dequant_i4_rows(const uint8_t *w, float *dst, int K, int group_size,
                     int row_begin, int row_end) {
    const int gpr = (K + group_size - 1) / group_size;      // 每行组数
    const int group_bytes = 4 + group_size / 2;             // 每组字节数
    const size_t row_bytes = static_cast<size_t>(gpr) * group_bytes;

#if defined(__aarch64__)
    const uint8x8_t m0F = vdup_n_u8(0x0F);
#endif

    for (int m = row_begin; m < row_end; ++m) {
        const uint8_t *row = w + static_cast<size_t>(m) * row_bytes;
        float *out = dst + static_cast<size_t>(m) * K;
        int col = 0;
        for (int g = 0; g < gpr; ++g) {
            const uint8_t *gp = row + static_cast<size_t>(g) * group_bytes;
            uint16_t sh, zh;
            std::memcpy(&sh, gp, 2);
            std::memcpy(&zh, gp + 2, 2);
            const float scale = half_bits_to_float_pf(sh);
            const float zero = half_bits_to_float_pf(zh);
            const uint8_t *pk = gp + 4;
            const int elems = (col + group_size <= K) ? group_size : (K - col);

#if defined(__aarch64__)
            // NEON：每次 8 字节 = 16 权重
            const float32x4_t zero_v = vdupq_n_f32(zero);
            int i = 0;
            for (; i + 16 <= elems; i += 16) {
                uint8x8_t raw = vld1_u8(pk + i / 2);
                uint8x8_t lo = vand_u8(raw, m0F);       // 偶数位权重
                uint8x8_t hi = vshr_n_u8(raw, 4);       // 奇数位权重
                uint8x8x2_t z = vzip_u8(lo, hi);        // 还原原始顺序
                // 两个半区各 8 权重：uint8 → uint16 → uint32 → f32
                // 注意：写到 out + col + i（col 是本组在行内的起始列），
                // 不是 out + i——否则每组都会覆盖行首。
                uint16x8_t u16_0 = vmovl_u8(z.val[0]);
                float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_0)));
                float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_0)));
                f0 = vmulq_n_f32(vsubq_f32(f0, zero_v), scale);
                f1 = vmulq_n_f32(vsubq_f32(f1, zero_v), scale);
                vst1q_f32(out + col + i, f0);
                vst1q_f32(out + col + i + 4, f1);
                uint16x8_t u16_1 = vmovl_u8(z.val[1]);
                float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_1)));
                float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_1)));
                f2 = vmulq_n_f32(vsubq_f32(f2, zero_v), scale);
                f3 = vmulq_n_f32(vsubq_f32(f3, zero_v), scale);
                vst1q_f32(out + col + i + 8, f2);
                vst1q_f32(out + col + i + 12, f3);
            }
            // 标量尾部
            for (; i < elems; ++i) {
                const int q = (i % 2 == 0) ? (pk[i / 2] & 0x0F) : ((pk[i / 2] >> 4) & 0x0F);
                out[col + i] = (static_cast<float>(q) - zero) * scale;
            }
#else
            // 纯标量兜底（非 aarch64 平台）
            for (int i = 0; i < elems; ++i) {
                const int q = (i % 2 == 0) ? (pk[i / 2] & 0x0F) : ((pk[i / 2] >> 4) & 0x0F);
                out[col + i] = (static_cast<float>(q) - zero) * scale;
            }
#endif
            col += group_size;
        }
    }
}

// ------------------------------------------------------------------------
// DequantPool: 常驻行切分线程池（原子行计数器，工作窃取式取块）
// ------------------------------------------------------------------------
struct DequantPool {
    const uint8_t *w = nullptr;
    float *dst = nullptr;
    int K = 0, group_size = 64;
    int total_rows = 0, chunk_rows = 1;
    std::atomic<int> next_row{0};             // 下一个未领取的行号

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    DequantPool() {
        const int p = default_parallelism_dq();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this] { worker_main(); });
        }
    }

    ~DequantPool() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }

    void worker_main() {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_dq(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_work();
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    // 动态取块：领一段行区间反量化，直到领完（快线程多做）
    void do_work() {
        for (;;) {
            const int begin = next_row.fetch_add(chunk_rows, std::memory_order_relaxed);
            if (begin >= total_rows) break;
            const int end = (begin + chunk_rows < total_rows) ? begin + chunk_rows
                                                                : total_rows;
            dequant_i4_rows(w, dst, K, group_size, begin, end);
        }
    }

    // fork-join 入口
    void run(const uint8_t *w_, float *dst_, int M, int K_, int group_size_) {
        w = w_; dst = dst_; K = K_; group_size = group_size_;
        total_rows = M;
        const int p = static_cast<int>(workers.size()) + 1;
        // 块大小：总行 / (4×核数)，下限 1（与 RowPool 同款再平衡粒度）
        chunk_rows = M / (p * 4);
        if (chunk_rows < 1) chunk_rows = 1;
        next_row.store(0, std::memory_order_relaxed);
        // release：上面的字段写入对看到 job_gen 的 worker 全部可见
        job_gen.store(++job_counter, std::memory_order_release);
        do_work();                            // master 同样参与取块
        expected_done += workers.size();
        spin_until_dq(done_gen, expected_done);
    }
};

// Meyers singleton：进程内唯一常驻池
DequantPool &dequant_pool() {
    static DequantPool p;
    return p;
}

} // namespace

// ========================================================================
// dequant_i4_to_f32: 按组反量化 [M,K] i4 权重到 fp32（对外暴露供单测）
// ========================================================================
// 大矩阵走常驻线程池（行间独立，工作窃取），小矩阵串行（省同步开销）。
// 并行只改变执行顺序，不改变任何一组的数值。
void dequant_i4_to_f32(const uint8_t *w, float *dst, int M, int K, int group_size) {
    if (static_cast<std::size_t>(M) * K >= kMinParallelElemsDq &&
        std::thread::hardware_concurrency() > 1) {
        dequant_pool().run(w, dst, M, K, group_size);
        return;
    }
    dequant_i4_rows(w, dst, K, group_size, 0, M);
}

namespace {

// fp16 权重 → fp32（逐元素，硬件/软件转换）
void convert_f16_to_f32(const uint16_t *w, float *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = half_bits_to_float_pf(w[i]);
}

} // namespace

// ========================================================================
// 融合 W4A8 批量 matmul：Y[M,N] = dequant(W_i4[M,K]) @ X[K,N]
// ========================================================================
// 替代"反量化到 fp32 + cblas_sgemm"：权重 i4 只读一遍、激活量化为 int8、
// 用 SDOT 一次算完全部 N 个 token，省去 fp32 往返（写 + 读 ≈ 8B/权重）的
// 访存。数值方案与 decode 的 sdot4 完全一致（W4A8），因此批量 prefill 与
// 逐 token decode 数值口径统一。仅 aarch64 + dotprod 可用，否则返回 false
// 由调用方回退 dequant+sgemm。
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
namespace {

// ------------------------------------------------------------------------
// unpack_group_i8: 把一组的 packed uint4 解包为 (q-8) 的 int8
// ------------------------------------------------------------------------
void unpack_group_i8(const uint8_t *pk, int8_t *w8, int elems) {
    int i = 0;
    const uint8x8_t m0F = vdup_n_u8(0x0F);
    const uint8x8_t v8 = vdup_n_u8(8);
    for (; i + 16 <= elems; i += 16) {
        uint8x8_t raw = vld1_u8(pk + i / 2);
        uint8x8_t lo = vand_u8(raw, m0F);       // 偶数位权重
        uint8x8_t hi = vshr_n_u8(raw, 4);       // 奇数位权重
        uint8x8x2_t z = vzip_u8(lo, hi);        // 还原原始顺序
        vst1_u8(reinterpret_cast<uint8_t *>(w8) + i, vsub_u8(z.val[0], v8));
        vst1_u8(reinterpret_cast<uint8_t *>(w8) + i + 8, vsub_u8(z.val[1], v8));
    }
    for (; i < elems; ++i) {
        const int q = (i % 2 == 0) ? (pk[i / 2] & 0x0F) : ((pk[i / 2] >> 4) & 0x0F);
        w8[i] = static_cast<int8_t>(q - 8);
    }
}

// ------------------------------------------------------------------------
// dot_i8: int8 点积（SDOT，16 元素/条；尾部标量）
// ------------------------------------------------------------------------
int dot_i8(const int8_t *a, const int8_t *b, int n) {
    int i = 0;
    int32x4_t d = vdupq_n_s32(0);
    for (; i + 16 <= n; i += 16) {
        d = vdotq_s32(d, vld1q_s8(a + i), vld1q_s8(b + i));
    }
    int acc = vaddvq_s32(d);
    for (; i < n; ++i) acc += static_cast<int>(a[i]) * static_cast<int>(b[i]);
    return acc;
}

// ------------------------------------------------------------------------
// matmul_i4_sdot_rows: 处理第 [row_begin, row_end) 行
// ------------------------------------------------------------------------
// 对每行：跨组累加；每组的权重解包一次后复用给全部 N 个 token。
// y 为 token 主序 [M, N]：y[m + n*M]。
void matmul_i4_sdot_rows(const uint8_t *w, const int8_t *xq, const float *ax_scale,
                         const int32_t *xqsum, float *y, int M, int K, int N,
                         int group_size, int row_begin, int row_end) {
    const int gpr = (K + group_size - 1) / group_size;
    const int group_bytes = 4 + group_size / 2;
    const size_t row_bytes = static_cast<size_t>(gpr) * group_bytes;
    std::vector<int8_t> w8(group_size);   // 一组解包后的权重（q-8）
    std::vector<float> acc(N);            // 当前行对 N 个 token 的累加

    for (int m = row_begin; m < row_end; ++m) {
        const uint8_t *row = w + static_cast<size_t>(m) * row_bytes;
        for (int n = 0; n < N; ++n) acc[n] = 0.0f;
        int col = 0;
        for (int g = 0; g < gpr; ++g) {
            const uint8_t *gp = row + static_cast<size_t>(g) * group_bytes;
            uint16_t sh, zh;
            std::memcpy(&sh, gp, 2);
            std::memcpy(&zh, gp + 2, 2);
            const float scale_w = half_bits_to_float_pf(sh);
            const float zero_m8 = half_bits_to_float_pf(zh) - 8.0f;
            const int group_elems = (col + group_size <= K) ? group_size : (K - col);
            unpack_group_i8(gp + 4, w8.data(), group_elems);
            // 本组对全部 N 个 token 做点积（权重解包一次，复用 N 次）
            for (int n = 0; n < N; ++n) {
                const int8_t *xq_gn = xq + static_cast<size_t>(n) * K + col;
                const int dot = dot_i8(w8.data(), xq_gn, group_elems);
                const float A = scale_w * ax_scale[n];
                const float xs = static_cast<float>(xqsum[static_cast<size_t>(n) * gpr + g]);
                acc[n] += A * static_cast<float>(dot) - (A * zero_m8) * xs;
            }
            col += group_size;
        }
        for (int n = 0; n < N; ++n) y[m + static_cast<size_t>(n) * M] = acc[n];
    }
}

// ------------------------------------------------------------------------
// MatmulI4Pool: 常驻行切分线程池（结构同 DequantPool）
// ------------------------------------------------------------------------
struct MatmulI4Pool {
    const uint8_t *w = nullptr;
    const int8_t *xq = nullptr;
    const float *ax_scale = nullptr;
    const int32_t *xqsum = nullptr;
    float *y = nullptr;
    int M = 0, K = 0, N = 0, group_size = 64;
    int total_rows = 0, chunk_rows = 1;
    std::atomic<int> next_row{0};

    std::atomic<std::uint64_t> job_gen{0};
    std::atomic<std::uint64_t> done_gen{0};
    std::atomic<bool> shutdown{false};
    std::vector<std::thread> workers;
    std::uint64_t job_counter = 0;
    std::uint64_t expected_done = 0;

    MatmulI4Pool() {
        const int p = default_parallelism_dq();
        workers.reserve(static_cast<size_t>(p - 1));
        for (int idx = 1; idx < p; ++idx) {
            workers.emplace_back([this] { worker_main(); });
        }
    }
    ~MatmulI4Pool() {
        shutdown.store(true, std::memory_order_release);
        job_gen.fetch_add(1, std::memory_order_release);
        for (auto &t : workers) t.join();
    }
    void worker_main() {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until_dq(job_gen, next_job);
            if (shutdown.load(std::memory_order_acquire)) return;
            do_work();
            done_gen.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }
    void do_work() {
        for (;;) {
            const int begin = next_row.fetch_add(chunk_rows, std::memory_order_relaxed);
            if (begin >= total_rows) break;
            const int end = (begin + chunk_rows < total_rows) ? begin + chunk_rows : total_rows;
            matmul_i4_sdot_rows(w, xq, ax_scale, xqsum, y, M, K, N, group_size, begin, end);
        }
    }
    void run(const uint8_t *w_, const int8_t *xq_, const float *ax_, const int32_t *xs_,
             float *y_, int M_, int K_, int N_, int gs_) {
        w = w_; xq = xq_; ax_scale = ax_; xqsum = xs_; y = y_;
        M = M_; K = K_; N = N_; group_size = gs_;
        total_rows = M;
        const int p = static_cast<int>(workers.size()) + 1;
        chunk_rows = M / (p * 4);
        if (chunk_rows < 1) chunk_rows = 1;
        next_row.store(0, std::memory_order_relaxed);
        job_gen.store(++job_counter, std::memory_order_release);
        do_work();
        expected_done += workers.size();
        spin_until_dq(done_gen, expected_done);
    }
};

MatmulI4Pool &matmul_i4_pool() {
    static MatmulI4Pool p;
    return p;
}

} // namespace
#endif // __aarch64__ && __ARM_FEATURE_DOTPROD

// ------------------------------------------------------------------------
// matmul_i4_batched: 量化激活 + 融合 matmul。返回 false 表示平台不支持。
// ------------------------------------------------------------------------
// xq / ax_scale / xqsum 为调用方（BatchPrefillBufs）提供的可复用 workspace。
bool matmul_i4_batched(const void *w, const float *x, float *y, int M, int K, int N,
                       int group_size, std::vector<int8_t> &xq,
                       std::vector<float> &ax_scale, std::vector<int32_t> &xqsum) {
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_DOTPROD)
    (void)w; (void)x; (void)y; (void)M; (void)K; (void)N; (void)group_size;
    (void)xq; (void)ax_scale; (void)xqsum;
    return false;
#else
    const int gpr = (K + group_size - 1) / group_size;
    // 1) 逐 token 对称 int8 量化 + 逐组激活和（与 sdot4 的 quantize_x 同语义）
    xq.resize(static_cast<size_t>(K) * N);
    ax_scale.resize(N);
    xqsum.resize(static_cast<size_t>(N) * gpr);
    for (int n = 0; n < N; ++n) {
        const float *xn = x + static_cast<size_t>(n) * K;
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float a = xn[k] < 0 ? -xn[k] : xn[k];
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        ax_scale[n] = scale;
        const float inv = 1.0f / scale;
        int8_t *xq_n = xq.data() + static_cast<size_t>(n) * K;
        int32_t *gsum = xqsum.data() + static_cast<size_t>(n) * gpr;
        for (int g = 0; g < gpr; ++g) gsum[g] = 0;
        for (int k = 0; k < K; ++k) {
            float v = xn[k] * inv;
            v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);
            const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);
            xq_n[k] = static_cast<int8_t>(q);
            gsum[k / group_size] += q;
        }
    }
    // 2) 行并行融合 matmul（小矩阵串行，省同步开销）
    if (static_cast<std::size_t>(M) * K >= kMinParallelElemsDq &&
        std::thread::hardware_concurrency() > 1) {
        matmul_i4_pool().run(static_cast<const uint8_t *>(w), xq.data(),
                             ax_scale.data(), xqsum.data(), y, M, K, N, group_size);
    } else {
        matmul_i4_sdot_rows(static_cast<const uint8_t *>(w), xq.data(),
                            ax_scale.data(), xqsum.data(), y, M, K, N, group_size, 0, M);
    }
    return true;
#endif
}

// ============================================================================
// QwenModel::forward_prefill_qwen35_batch() — 批量 prefill 主入口
// ============================================================================
// 返回值：最后一个 prompt token 的 greedy 下一 token；
//         -2 表示本实现无法处理（平台无 GEMM 后端 / dtype 不支持），
//         调用方负责回退逐 token 路径。
// ============================================================================
int QwenModel::forward_prefill_qwen35_batch(const int *token_ids, int n,
                                            TopKResult *topk, int topk_k) {
#if !defined(TINYQWEN_HAS_CBLAS)
    // 无 BLAS 后端（如 Android）：交给调用方回退逐 token
    (void)token_ids; (void)n; (void)topk; (void)topk_k;
    return -2;
#else
    const int hidden = static_cast<int>(cfg_.hidden_size);
    const int inter = static_cast<int>(cfg_.intermediate_size);
    const int vocab = static_cast<int>(cfg_.vocab_size);
    const int n_heads = static_cast<int>(cfg_.n_heads);
    const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim);
    const int base_pos = kv_.seq_len();
    const size_t N = static_cast<size_t>(n);

    // dtype 支持：i4（反量化）/ f32（直用）/ f16（转换）
    if (dtype_ != Dtype::kI4 && dtype_ != Dtype::kF32 && dtype_ != Dtype::kF16)
        return -2;

    // 容量检查（与逐 token 路径同样的 fail-loud 语义）
    if (base_pos + n > max_seq_len_) {
        std::fprintf(stderr,
                     "tinyqwen: batch prefill %d+%d exceeds max_seq_len %d\n",
                     base_pos, n, max_seq_len_);
        std::abort();
    }

    Profiler &prof = *profiler_;
    char name[64];
    const auto scope = [&](const char *fmt, int layer) {
        std::snprintf(name, sizeof(name), fmt, layer);
        return name;
    };

    // ---- workspace 按需扩容（token 主序 [dim, N]）----
    auto &bp = bp_;
    bp.hid.resize(static_cast<size_t>(hidden) * N);
    bp.normed.resize(static_cast<size_t>(hidden) * N);
    bp.out.resize(static_cast<size_t>(hidden) * N);   // o_proj / ffn 输出复用
    const int conv_dim = gdn_conv_dim_;
    const int value_dim = gdn_value_dim_;
    const int n_v_heads = static_cast<int>(cfg_.linear_num_v_heads);
    const int n_qk_heads = static_cast<int>(cfg_.linear_num_qk_heads);
    const int qk_hd = static_cast<int>(cfg_.linear_qk_head_dim);
    const int v_hd = static_cast<int>(cfg_.linear_v_head_dim);
    const int key_dim = gdn_qk_dim_;
    bp.mixed.resize(static_cast<size_t>(conv_dim) * N);
    bp.z.resize(static_cast<size_t>(value_dim) * N);
    bp.b.resize(static_cast<size_t>(n_v_heads) * N);
    bp.a.resize(static_cast<size_t>(n_v_heads) * N);
    bp.gdn_out.resize(static_cast<size_t>(value_dim) * N);
    bp.q_full.resize(static_cast<size_t>(2) * q_dim_ * N);
    bp.k.resize(static_cast<size_t>(kv_dim_) * N);
    bp.v.resize(static_cast<size_t>(kv_dim_) * N);
    bp.attn.resize(static_cast<size_t>(q_dim_) * N);
    bp.gate.resize(static_cast<size_t>(inter) * N);
    bp.up.resize(static_cast<size_t>(inter) * N);

    // GEMM 前的权重准备：把权重变成 fp32（i4 反量化 / f16 转换 / f32 直用），
    // 返回参与 GEMM 的指针（可能指向 bp.deq，也可能是权重本体）。
    const auto prep_w = [&](const void *w, int M, int K) -> const float * {
        if (dtype_ == Dtype::kF32) return static_cast<const float *>(w);
        bp.deq.resize(static_cast<size_t>(M) * K);
        if (dtype_ == Dtype::kI4) {
            dequant_i4_to_f32(static_cast<const uint8_t *>(w), bp.deq.data(),
                              M, K, group_size_);
        } else {
            convert_f16_to_f32(static_cast<const uint16_t *>(w), bp.deq.data(),
                               static_cast<size_t>(M) * K);
        }
        return bp.deq.data();
    };
    // GEMM 包装：Y = W @ X（token 主序），失败即回退
    // 实验开关：融合 W4A8 matmul 默认**关闭**——实测它比 dequant+AMX-sgemm
    // 慢（4B-61tok 0.70×：手写 NEON SDOT 干不过 AMX，省下的访存填不平算力
    // 差距，见优化日志"证伪归档"）。设置 TINYQWEN_FUSED_MM 可强制启用对照。
    const bool use_fused_mm = std::getenv("TINYQWEN_FUSED_MM") != nullptr;
    const auto do_gemm = [&](const void *w, const float *x, float *y,
                             int M, int K) -> bool {
        if (use_fused_mm && dtype_ == Dtype::kI4 &&
            matmul_i4_batched(w, x, y, M, K, n, group_size_,
                              bp.xq, bp.ax_scale, bp.xqsum)) {
            return true;
        }
        const float *wf = prep_w(w, M, K);
        return gemm_wx(wf, x, y, M, K, n);
    };

    // =====================================================================
    // Step 1: 词嵌入（逐列填 hid_batch）
    // =====================================================================
    {
        ScopedTimer t(prof, "prefill_embed");
        for (int c = 0; c < n; ++c) {
            const int tid = token_ids[c];
            float *dst = bp.hid.data() + static_cast<size_t>(c) * hidden;
            if (dtype_ == Dtype::kF32 || dtype_ == Dtype::kI4) {
                std::memcpy(dst, static_cast<const float *>(embed_) +
                                         static_cast<size_t>(tid) * hidden,
                            hidden * sizeof(float));
            } else {
                const uint16_t *row = static_cast<const uint16_t *>(embed_) +
                                      static_cast<size_t>(tid) * hidden;
                for (int j = 0; j < hidden; ++j) dst[j] = half_to_float(row[j]);
            }
        }
    }

    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // =====================================================================
    // Step 2: 逐层（投影批量 GEMM，跨 token 耦合部分顺序扫描）
    // =====================================================================
    for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
        const LayerWeights &w = layers_[i];

        // ---- input norm（逐列）----
        for (int c = 0; c < n; ++c) {
            backend_->rmsnorm(bp.hid.data() + static_cast<size_t>(c) * hidden,
                              w.input_ln,
                              bp.normed.data() + static_cast<size_t>(c) * hidden,
                              hidden, cfg_.rms_norm_eps);
        }

        if (cfg_.is_linear_layer(i)) {
            // =============== GDN 层 ===============
            const int li = cfg_.linear_layer_cache_index(i);

            // 投影：mixed/z/b/a = W @ normed（批量 GEMM）
            {
                ScopedTimer t(prof, scope("layer_%d.bp_gdn_proj", i));
                if (!do_gemm(w.gdn_in_qkv, bp.normed.data(), bp.mixed.data(), conv_dim, hidden))
                    return -2;
                if (!do_gemm(w.gdn_in_z, bp.normed.data(), bp.z.data(), value_dim, hidden))
                    return -2;
                if (!do_gemm(w.gdn_in_b, bp.normed.data(), bp.b.data(), n_v_heads, hidden))
                    return -2;
                if (!do_gemm(w.gdn_in_a, bp.normed.data(), bp.a.data(), n_v_heads, hidden))
                    return -2;
            }

            // 顺序扫描：conv1d + l2norm + delta rule + 门控 norm
            {
                ScopedTimer t(prof, scope("layer_%d.bp_gdn_scan", i));
                const float q_scale = 1.0f / std::sqrt(static_cast<float>(qk_hd));
                const int rep = n_v_heads / n_qk_heads;
                for (int c = 0; c < n; ++c) {
                    float *mixed = bp.mixed.data() + static_cast<size_t>(c) * conv_dim;
                    const float *zc = bp.z.data() + static_cast<size_t>(c) * value_dim;
                    const float *bc = bp.b.data() + static_cast<size_t>(c) * n_v_heads;
                    const float *ac = bp.a.data() + static_cast<size_t>(c) * n_v_heads;
                    float *outc = bp.gdn_out.data() + static_cast<size_t>(c) * value_dim;

                    // conv1d 单步（就地，含 silu；状态推进）
                    backend_->causal_conv1d_update(mixed, gdn_state_.conv(li),
                                                   w.gdn_conv_w, mixed, conv_dim,
                                                   static_cast<int>(cfg_.linear_conv_kernel_dim));

                    // q/k 逐头 l2norm + q 缩放（与 forward_token 同序）
                    for (int h = 0; h < n_qk_heads; ++h) {
                        float *qh = mixed + h * qk_hd;
                        float *kh = mixed + key_dim + h * qk_hd;
                        backend_->l2norm_inplace(qh, qk_hd, 1e-6f);
                        for (int d = 0; d < qk_hd; ++d) qh[d] *= q_scale;
                        backend_->l2norm_inplace(kh, qk_hd, 1e-6f);
                    }

                    // gated delta rule 递归（状态矩阵逐头更新）
                    float *S = gdn_state_.recurrent(li);
                    for (int h = 0; h < n_v_heads; ++h) {
                        const int qk_h = h / rep;
                        const float *qh = mixed + qk_h * qk_hd;
                        const float *kh = mixed + key_dim + qk_h * qk_hd;
                        const float *vh = mixed + 2 * key_dim + h * v_hd;
                        const float g = -std::exp(w.gdn_a_log[h]) *
                                        softplusf32(ac[h] + w.gdn_dt_bias[h]);
                        const float beta = sigmoidf32(bc[h]);
                        backend_->gdn_step(S + static_cast<size_t>(h) * qk_hd * v_hd,
                                           qh, kh, vh, g, beta, outc + h * v_hd,
                                           qk_hd, v_hd);
                    }

                    // 门控 RMSNorm（按 v 头）
                    for (int h = 0; h < n_v_heads; ++h) {
                        backend_->rmsnorm_gated(outc + h * v_hd, zc + h * v_hd,
                                                w.gdn_norm, outc + h * v_hd, v_hd,
                                                cfg_.rms_norm_eps);
                    }
                }
            }

            // out_proj（批量 GEMM）+ 残差
            {
                ScopedTimer t(prof, scope("layer_%d.bp_gdn_out", i));
                if (!do_gemm(w.gdn_out_proj, bp.gdn_out.data(), bp.out.data(), hidden, value_dim))
                    return -2;
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    bp.hid.data()[j] += bp.out.data()[j];
            }

        } else {
            // =============== Full attention 层 ===============
            const int ci = cfg_.full_layer_cache_index(i);

            // q(+gate)/k/v 投影（批量 GEMM）
            {
                ScopedTimer t(prof, scope("layer_%d.bp_qkv", i));
                if (!do_gemm(w.q_proj, bp.normed.data(), bp.q_full.data(), 2 * q_dim_, hidden))
                    return -2;
                if (!do_gemm(w.k_proj, bp.normed.data(), bp.k.data(), kv_dim_, hidden))
                    return -2;
                if (!do_gemm(w.v_proj, bp.normed.data(), bp.v.data(), kv_dim_, hidden))
                    return -2;
            }

            // 顺序扫描：解交错 + QK-norm + partial RoPE + KV 追加 + attention + 门
            {
                ScopedTimer t(prof, scope("layer_%d.bp_attn_scan", i));
                for (int c = 0; c < n; ++c) {
                    const int pos = base_pos + c;
                    const float *qf = bp.q_full.data() + static_cast<size_t>(c) * 2 * q_dim_;
                    float *kc = bp.k.data() + static_cast<size_t>(c) * kv_dim_;
                    float *vc = bp.v.data() + static_cast<size_t>(c) * kv_dim_;

                    // 按 head 解交错：前半 query、后半输出门（复用单 token workspace）
                    for (int h = 0; h < n_heads; ++h) {
                        const float *src = qf + h * 2 * head_dim;
                        std::memcpy(q_.data() + h * head_dim, src, head_dim * sizeof(float));
                        std::memcpy(q_gate_.data() + h * head_dim, src + head_dim,
                                    head_dim * sizeof(float));
                    }

                    // QK per-head RMSNorm（复用 k_ workspace）
                    std::memcpy(k_.data(), kc, kv_dim_ * sizeof(float));
                    for (int h = 0; h < n_heads; ++h)
                        backend_->rmsnorm(q_.data() + h * head_dim, w.q_norm,
                                          q_.data() + h * head_dim, head_dim, cfg_.rms_norm_eps);
                    for (int h = 0; h < n_kv_heads; ++h)
                        backend_->rmsnorm(k_.data() + h * head_dim, w.k_norm,
                                          k_.data() + h * head_dim, head_dim, cfg_.rms_norm_eps);

                    // partial RoPE（位置 = base_pos + c）
                    backend_->partial_rope(q_.data(), k_.data(), n_heads, n_kv_heads,
                                           head_dim, rotary_dim_, pos, cfg_.rope_theta);

                    // KV 追加到 pos（统一写入入口，内部按精度转换）
                    kv_.write_token(ci, pos, k_.data(), vc);

                    // attention（因果：看到 [0, pos]）+ sigmoid 输出门
                    //   attention_kv 按 KV 精度分发（fp32 / fp16-KV 融合）
                    attention_kv(q_.data(), ci, pos + 1, n_heads, n_kv_heads, head_dim,
                                 attn_scale, attn_.data());
                    float *atc = bp.attn.data() + static_cast<size_t>(c) * q_dim_;
                    for (int j = 0; j < q_dim_; ++j)
                        atc[j] = attn_[j] * sigmoidf32(q_gate_[j]);
                }
            }

            // o_proj（批量 GEMM）+ 残差
            {
                ScopedTimer t(prof, scope("layer_%d.bp_o", i));
                if (!do_gemm(w.o_proj, bp.attn.data(), bp.out.data(), hidden, q_dim_))
                    return -2;
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    bp.hid.data()[j] += bp.out.data()[j];
            }
        }

        // =============== FFN（全批量） ===============
        {
            ScopedTimer t(prof, scope("layer_%d.bp_ffn", i));
            for (int c = 0; c < n; ++c) {
                backend_->rmsnorm(bp.hid.data() + static_cast<size_t>(c) * hidden,
                                  w.post_ln,
                                  bp.normed.data() + static_cast<size_t>(c) * hidden,
                                  hidden, cfg_.rms_norm_eps);
            }
            if (!do_gemm(w.gate, bp.normed.data(), bp.gate.data(), inter, hidden))
                return -2;
            if (!do_gemm(w.up, bp.normed.data(), bp.up.data(), inter, hidden))
                return -2;
            for (int c = 0; c < n; ++c) {
                backend_->swiglu(bp.gate.data() + static_cast<size_t>(c) * inter,
                                 bp.up.data() + static_cast<size_t>(c) * inter, inter);
            }
            if (!do_gemm(w.down, bp.gate.data(), bp.out.data(), hidden, inter))
                return -2;
            for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                bp.hid.data()[j] += bp.out.data()[j];
        }
    }

    // =====================================================================
    // Step 3 + 4: 只对最后一个 token 算 final norm + lm_head + argmax
    // =====================================================================
    int next = -1;
    {
        ScopedTimer t(prof, "final_norm");
        backend_->rmsnorm(bp.hid.data() + static_cast<size_t>(n - 1) * hidden,
                          final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
    }
    {
        ScopedTimer t(prof, "lm_head");
        if (lm_head_is_f32_) {
            matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                       logits_.data(), vocab, hidden);
        } else {
            mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
        }
    }
    {
        ScopedTimer t(prof, "topk_argmax");
        if (topk) {
            top_k_logits_pf(logits_.data(), vocab, topk_k, topk);
            next = topk->indices.empty() ? 0 : topk->indices[0];
        } else {
            next = argmax(logits_.data(), vocab);
        }
    }

    // 批量提交：KV 长度 +n，token 计数 +n
    kv_.advance(n);
    token_count_ += n;
    return next;
#endif // TINYQWEN_HAS_CBLAS
}

} // namespace tinyqwen
