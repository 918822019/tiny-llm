// ============================================================================
// test_matvec_i4.cpp — INT4 matvec / matmul 单元测试
// ============================================================================
// 本文件验证 INT4 量化矩阵向量乘法的正确性，覆盖所有实现变体：
//   - ref      — 标量参考实现（double 累加）
//   - neon     — ARM NEON SIMD 单线程
//   - neon_mt  — ARM NEON + 多线程行切分
//   - sdot     — ARM dotprod 扩展指令（W4A8 SDOT）
//   - sdot_mt  — SDOT + 多线程
//   - sdot2 / sdot2_mt — 预计算缓存 + 2-row 并行
//   - sdot3 / sdot3_mt — work-stealing + 内联组头 + 128 位解包
//   - sdot4 / sdot4_mt — sdot3 + 修正组头转换守卫（硬件 _Float16 FCVT）
//
// 背景：i4 kernel 此前只有 ref 注释约束、没有自动化验证。量化导出（RTN/HQQ）
// 与 kernel 之间的契约（interleaved 布局、低 nibble 在前、(q-zero)×scale 反量化、
// fp16 scale/zero）全靠这些测试把关。
//
// 测试方法论：
//   1. pack_i4_rtn()       — 测试侧独立实现 RTN 量化 + interleaved packing
//   2. naive_matvec_i4()   — 测试侧独立实现反量化 + double 累加 matvec
//   3. matvec_i4_w4a8_naive() — W4A8 SDOT 方案的朴素参照
//   4. check_matvec_i4_impl() — 将 kernel 输出与上述独立实现对比
//
// kernel 函数注册在匿名 namespace，测试统一走 dispatch（set_matvec_i4_impl_by_name）。
// ============================================================================

#include "test_framework.h"  // 自研测试框架

#include "dispatch.h"   // set_matvec_i4_impl_by_name / matvec_i4 / matmul_i4
#include "ref_ops.h"    // float_to_half / half_to_float
#include "qwen_model.h" // tinyqwen::dequant_i4_to_f32（批量 prefill 的反量化，回归对照）

#include <algorithm>    // std::min, std::max
#include <cmath>        // std::fabs, std::lround
#include <cstdint>      // uint8_t, uint16_t
#include <cstring>      // std::memcpy
#include <random>       // std::mt19937, std::normal_distribution
#include <vector>       // std::vector

namespace {

using tinyqwen::float_to_half;   // float → fp16 编码
using tinyqwen::half_to_float;   // fp16 编码 → float

// interleaved i4 布局中每个 group 的头部大小：scale(fp16) + zero(fp16) = 4 bytes
constexpr int kHdr = 4;

// =========================================================================
// pack_i4_rtn() — RTN 量化并 pack 成 interleaved i4 字节
// =========================================================================
// 镜像 exporter 的 quantize_tensor_i4 逻辑，在测试侧独立实现。
//
// interleaved 布局（per group）：
//   [scale_h: 2B][zero_h: 2B][packed_nibbles: group_size/2 B]
//   每个字节存两个 4-bit 量化值：低 nibble = 偶数索引元素，高 nibble = 奇数索引元素
//
// 参数：
//   w          — f32 权重矩阵（row-major, out_dim × in_dim）
//   out_dim    — 输出维度（行数）
//   in_dim     — 输入维度（列数）
//   group_size — 量化分组大小（如 64 或 128）
// 返回：packed 字节数组
std::vector<uint8_t> pack_i4_rtn(const std::vector<float> &w, int out_dim, int in_dim,
                                 int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;  // 每行的组数
    const int group_total = kHdr + group_size / 2;                       // 每组总字节数
    const int row_bytes = groups_per_row * group_total;                  // 每行总字节数
    std::vector<uint8_t> packed(static_cast<size_t>(out_dim) * row_bytes, 0);

    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            const int start = g * group_size;                             // 组起始列
            const int end = std::min(start + group_size, in_dim);         // 组结束列（处理尾部）
            // RTN：找组内 min/max 确定量化范围
            float vmin = w[o * in_dim + start], vmax = vmin;
            for (int i = start; i < end; ++i) {
                vmin = std::min(vmin, w[o * in_dim + i]);
                vmax = std::max(vmax, w[o * in_dim + i]);
            }
            // 计算 scale 和 zero：w ≈ (q - zero) * scale, q ∈ [0, 15]
            const float scale = (vmax > vmin) ? (vmax - vmin) / 15.0f : 0.0f;
            const float zero = (scale > 0) ? -vmin / scale : 0.0f;
            // 转为 fp16 编码（存储格式）
            const uint16_t scale_h = float_to_half(scale);
            const uint16_t zero_h = float_to_half(zero);
            // 与 C++ kernel 对齐：用 fp16 往返后的 s/z 算 q（模拟精度损失）
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            // 写入 group header
            uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            std::memcpy(gp, &scale_h, 2);           // scale (fp16)
            std::memcpy(gp + 2, &zero_h, 2);        // zero (fp16)
            // 量化并 pack nibbles
            for (int i = start; i < end; ++i) {
                int q = (s > 0) ? static_cast<int>(std::lround(w[o * in_dim + i] / s + z)) : 0;
                q = std::max(0, std::min(15, q));   // clamp 到 [0, 15]
                const int li = i - start;            // 组内局部索引
                if (li % 2 == 0) gp[kHdr + li / 2] = static_cast<uint8_t>(q);        // 低 nibble
                else gp[kHdr + li / 2] |= static_cast<uint8_t>(q << 4);              // 高 nibble
            }
        }
    }
    return packed;
}

// =========================================================================
// naive_matvec_i4() — 测试侧独立的反量化 + matvec（double 累加）
// =========================================================================
// 从 packed 字节逐元素反量化后做矩阵向量乘，用 double 累加保证精度。
// 作为 ref/neon/sdot 等所有 kernel 的统一数值锚点。
std::vector<float> naive_matvec_i4(const std::vector<uint8_t> &packed,
                                   const std::vector<float> &x,
                                   int out_dim, int in_dim, int group_size) {
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        double acc = 0.0;  // double 累加
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            // 读取 scale 和 zero
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            // 逐元素反量化并累加
            for (int i = start; i < end; ++i) {
                const int li = i - start;
                const uint8_t byte = gp[kHdr + li / 2];
                const int q = (li % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);  // 解包 nibble
                acc += static_cast<double>((q - z) * s) * x[i];  // 反量化 × 输入
            }
        }
        y[o] = static_cast<float>(acc);
    }
    return y;
}

// =========================================================================
// random_vec() — 生成随机向量（正态分布）
// =========================================================================
std::vector<float> random_vec(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);                              // Mersenne Twister RNG
    std::normal_distribution<float> dist(0.0f, 1.0f);    // N(0,1) 标准正态
    std::vector<float> v(n);
    for (auto &x: v) x = dist(rng);
    return v;
}

// =========================================================================
// check_matvec_i4_impl() — 通用 matvec_i4 正确性检查函数
// =========================================================================
// 将指定 impl 的 kernel 输出与 naive_matvec_i4 对比。
// NEON 系实现在非 aarch64 平台自动跳过（不算失败）。
void check_matvec_i4_impl(const char *impl, int out_dim, int in_dim, int group_size) {
    if (!tinyqwen::set_matvec_i4_impl_by_name(impl)) {
        // 非 aarch64 平台没有 neon 系实现——跳过而不是失败
        if (std::string(impl).rfind("neon", 0) == 0) return;
        TQ_FAIL(std::string("set_matvec_i4_impl_by_name failed: ") + impl);
    }
    // 生成随机权重和输入
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 42);
    const std::vector<float> x = random_vec(in_dim, 7);
    // RTN 量化 + pack
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    // 执行被测 kernel
    std::vector<float> y(out_dim);
    tinyqwen::matvec_i4(packed.data(), x.data(), y.data(), out_dim, in_dim, group_size);
    // 与独立实现对比
    const std::vector<float> ref = naive_matvec_i4(packed, x, out_dim, in_dim, group_size);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    // ref kernel 用 double 累加、naive 也是 double：差异只来自 float 舍入
    if (max_err >= 1e-3) {
        TQ_FAIL("matvec_i4(" + std::string(impl) + ", g=" + std::to_string(group_size) +
                ") max_err=" + std::to_string(max_err));
    }
}

} // namespace

// ---------------------------------------------------------------------------
// 测试：手工小例验证 interleaved i4 布局约定
//
// 1 行 × 4 元素，group_size=4（1 组）：
//   q = [1, 2, 15, 0], scale=2.0, zero=1.0
//   反量化：dequant[i] = (q[i] - zero) * scale = [0, 2, 28, -2]
//   x = [1,1,1,1] → y = 0+2+28-2 = 28
//
// 手动构造 packed 字节：
//   byte0 = (q[1]<<4)|q[0] = (2<<4)|1 = 0x21
//   byte1 = (q[3]<<4)|q[2] = (0<<4)|15 = 0x0F
// ---------------------------------------------------------------------------
TEST(matvec_i4_handcrafted_layout) {
    const uint16_t scale_h = float_to_half(2.0f);   // scale = 2.0 → fp16
    const uint16_t zero_h = float_to_half(1.0f);     // zero = 1.0 → fp16
    std::vector<uint8_t> packed(kHdr + 2, 0);        // 4B header + 2B nibbles
    std::memcpy(packed.data(), &scale_h, 2);          // 写入 scale
    std::memcpy(packed.data() + 2, &zero_h, 2);       // 写入 zero
    packed[kHdr] = (2 << 4) | 1;                      // byte0: 元素1=2(高), 元素0=1(低)
    packed[kHdr + 1] = (0 << 4) | 15;                 // byte1: 元素3=0(高), 元素2=15(低)
    const float x[4] = {1.0f, 1.0f, 1.0f, 1.0f};     // 全 1 输入
    float y = 0.0f;
    tinyqwen::set_matvec_i4_impl_by_name("ref");       // 使用 ref 实现
    tinyqwen::matvec_i4(packed.data(), x, &y, 1, 4, 4);
    EXPECT_NEAR(y, 28.0f, 1e-5);                       // 期望 0+2+28-2 = 28
}

// ---------------------------------------------------------------------------
// ref kernel vs naive 独立实现（随机权重）
// ---------------------------------------------------------------------------
TEST(matvec_i4_ref_matches_naive_g128) { check_matvec_i4_impl("ref", 64, 256, 128); }  // group=128
TEST(matvec_i4_ref_matches_naive_g64) { check_matvec_i4_impl("ref", 64, 256, 64); }    // group=64

// ---------------------------------------------------------------------------
// in_dim 非 group 整数倍（尾部组不完整）也要对
// ---------------------------------------------------------------------------
TEST(matvec_i4_ref_partial_group) { check_matvec_i4_impl("ref", 32, 200, 64); }  // 200/64=3余8

// ---------------------------------------------------------------------------
// NEON kernel 与 naive 对齐（仅 aarch64 有注册；其他平台自动跳过）
// ---------------------------------------------------------------------------
TEST(matvec_i4_neon_matches_naive_g128) { check_matvec_i4_impl("neon", 96, 384, 128); }
TEST(matvec_i4_neon_matches_naive_g64) { check_matvec_i4_impl("neon", 96, 384, 64); }

// ---------------------------------------------------------------------------
// NEON + 多线程：行切分不改变数值，结果必须与单线程一致
// 用 > kMinParallelElems（262144）的矩阵确保真的走多线程路径
// ---------------------------------------------------------------------------
TEST(matvec_i4_neon_mt_matches_naive_g64) { check_matvec_i4_impl("neon_mt", 512, 896, 64); }
TEST(matvec_i4_neon_mt_matches_naive_g128) { check_matvec_i4_impl("neon_mt", 512, 1024, 128); }

// =========================================================================
// W4A8 SDOT 朴素参考实现
// =========================================================================
// 与 kernel 同一套量化方案（激活对称 int8 + 非对称 zero 修正），
// 用纯标量整数点积实现。用来核对 NEON SDOT kernel 的正确性。
// 两者同方案，应几乎逐位一致（只有 fp32 加法顺序差）。
//
// 数学推导：
//   w_dequant = (q_w - zero_w) * scale_w
//   x_quant = round(x / scale_x)，对称量化到 [-127, 127]
//   dot = sum((q_w - 8) * x_q)  （q_w 偏移到有符号 [-8,7]）
//   result = A * dot - C * sum(x_q)
//   其中 A = scale_w * scale_x, C = A * (zero_w - 8)
std::vector<float> matvec_i4_w4a8_naive(const std::vector<uint8_t> &packed,
                                        const std::vector<float> &x,
                                        int out_dim, int in_dim, int group_size) {
    // ---- 激活对称 int8 量化（与 kernel 的 quantize_x_i8 逐字一致）----
    float amax = 0.0f;
    for (int i = 0; i < in_dim; ++i) {
        const float a = x[i] < 0 ? -x[i] : x[i];  // abs
        if (a > amax) amax = a;
    }
    const float scale_x = amax > 0.0f ? amax / 127.0f : 1.0f;  // 对称量化 scale
    const float inv = 1.0f / scale_x;                            // 逆 scale
    std::vector<int> xq(in_dim);                                  // 量化后的激活
    std::vector<int> prefix(in_dim + 1, 0);                       // 前缀和（加速 sum(xq)）
    for (int i = 0; i < in_dim; ++i) {
        float v = x[i] * inv;
        v = v > 127.0f ? 127.0f : (v < -127.0f ? -127.0f : v);  // clamp
        const int q = static_cast<int>(v >= 0 ? v + 0.5f : v - 0.5f);  // round
        xq[i] = q;
        prefix[i + 1] = prefix[i] + q;  // 前缀和
    }

    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    std::vector<float> y(out_dim);
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = packed.data() + static_cast<size_t>(o) * row_bytes;
        float acc = 0.0f;
        int col = 0;
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = row + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float scale_w = tinyqwen::half_to_float(scale_h);
            const float zero_w = tinyqwen::half_to_float(zero_h);
            const float A = scale_w * scale_x;             // 组合缩放因子
            const float C = A * (zero_w - 8.0f);           // zero 修正项
            const uint8_t *pk = gp + kHdr;                  // packed nibbles 起始
            const int elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);  // 尾部处理
            // 整数点积：sum((q_w - 8) * x_q)
            int dot_g = 0;
            for (int i = 0; i < elems; ++i) {
                const int q = (i % 2 == 0) ? (pk[i / 2] & 0x0F) : ((pk[i / 2] >> 4) & 0x0F);
                dot_g += (q - 8) * xq[col + i];
            }
            // 利用前缀和快速计算 sum(xq[col..col+elems))
            const int xqsum = prefix[col + elems] - prefix[col];
            acc += A * static_cast<float>(dot_g) - C * static_cast<float>(xqsum);
            col += group_size;
        }
        y[o] = acc;
    }
    return y;
}

// =========================================================================
// check_matvec_i4_sdot_impl() — SDOT kernel 正确性检查
// =========================================================================
// 与 W4A8 朴素参考对比，容差 1e-2（SDOT 整数运算链更长，累积误差略大）。
void check_matvec_i4_sdot_impl(const char *impl, int out_dim, int in_dim, int group_size) {
    if (!tinyqwen::set_matvec_i4_impl_by_name(impl)) {
        return; // 平台无 dotprod 扩展，跳过
    }
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 42);
    const std::vector<float> x = random_vec(in_dim, 7);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> y(out_dim);
    tinyqwen::matvec_i4(packed.data(), x.data(), y.data(), out_dim, in_dim, group_size);
    const std::vector<float> ref = matvec_i4_w4a8_naive(packed, x, out_dim, in_dim, group_size);
    double max_err = 0.0;
    for (int i = 0; i < out_dim; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(y[i]) - ref[i]));
    }
    if (max_err >= 1e-2) {
        TQ_FAIL(std::string("matvec_i4(") + impl + ", g=" + std::to_string(group_size) +
                ") max_err=" + std::to_string(max_err));
    }
}

// 便捷包装：默认使用 "sdot" 实现
void check_matvec_i4_sdot(int out_dim, int in_dim, int group_size) {
    check_matvec_i4_sdot_impl("sdot", out_dim, in_dim, group_size);
}

// ---------------------------------------------------------------------------
// SDOT kernel vs W4A8 朴素参考
// ---------------------------------------------------------------------------
TEST(matvec_i4_sdot_matches_w4a8_naive_g64) { check_matvec_i4_sdot(96, 384, 64); }
TEST(matvec_i4_sdot_matches_w4a8_naive_g128) { check_matvec_i4_sdot(96, 384, 128); }
TEST(matvec_i4_sdot_partial) { check_matvec_i4_sdot(32, 200, 64); }  // 尾部组

// SDOT + 多线程
TEST(matvec_i4_sdot_mt_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot_mt", 512, 896, 64); }
TEST(matvec_i4_sdot_mt_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot_mt", 512, 1024, 128); }

// ---------------------------------------------------------------------------
// SDOT v3（动态调度 + 内联组头 + 128 位解包）
// ---------------------------------------------------------------------------
TEST(matvec_i4_sdot3_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot3", 96, 384, 64); }
TEST(matvec_i4_sdot3_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot3", 96, 384, 128); }
TEST(matvec_i4_sdot3_partial) { check_matvec_i4_sdot_impl("sdot3", 32, 200, 64); }  // 尾部组
TEST(matvec_i4_sdot3_mt_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot3_mt", 512, 896, 64); }
TEST(matvec_i4_sdot3_mt_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot3_mt", 512, 1024, 128); }
// 组内非 32 倍数边界（48 = 32 主块 + 16 兜底块）+ 尾部组恰好 16 元素
// （544 = 48×11 + 16）；512×544 > 并行阈值 262144，真正走线程池路径
TEST(matvec_i4_sdot3_mt_matches_w4a8_naive_g48) { check_matvec_i4_sdot_impl("sdot3_mt", 512, 544, 48); }

// ---------------------------------------------------------------------------
// sdot3 vs sdot2 逐位一致：两项改动（内联组头/128 位解包）都不改变数值——
// fp16→fp32 无损、整数点积与累加顺序无关、浮点还原同序。任何偏差都说明
// 实现有 bug，因此容差为 0（逐位比较）。
// ---------------------------------------------------------------------------
TEST(matvec_i4_sdot3_bitexact_vs_sdot2) {
    if (!tinyqwen::set_matvec_i4_impl_by_name("sdot2")) return;  // 平台无 dotprod
    const int out_dim = 128, in_dim = 896, group_size = 64;
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 4242);
    const std::vector<float> x = random_vec(in_dim, 4243);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> y2(out_dim), y3(out_dim);
    tinyqwen::set_matvec_i4_impl_by_name("sdot2");
    tinyqwen::matvec_i4(packed.data(), x.data(), y2.data(), out_dim, in_dim, group_size);
    tinyqwen::set_matvec_i4_impl_by_name("sdot3");
    tinyqwen::matvec_i4(packed.data(), x.data(), y3.data(), out_dim, in_dim, group_size);
    for (int i = 0; i < out_dim; ++i) {
        if (y2[i] != y3[i]) {  // 逐位比较（不是近似）
            TQ_FAIL("sdot3 vs sdot2 非逐位一致 @ row " + std::to_string(i) +
                    ": sdot2=" + std::to_string(y2[i]) + " sdot3=" + std::to_string(y3[i]));
            return;
        }
    }
    tinyqwen::set_matvec_i4_impl_by_name("ref");  // 复位，避免影响后续测试
}

// ---------------------------------------------------------------------------
// SDOT v4（修正组头转换特性守卫：_Float16 硬件 FCVT）
// ---------------------------------------------------------------------------
TEST(matvec_i4_sdot4_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot4", 96, 384, 64); }
TEST(matvec_i4_sdot4_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot4", 96, 384, 128); }
TEST(matvec_i4_sdot4_partial) { check_matvec_i4_sdot_impl("sdot4", 32, 200, 64); }  // 尾部组
TEST(matvec_i4_sdot4_mt_matches_w4a8_naive_g64) { check_matvec_i4_sdot_impl("sdot4_mt", 512, 896, 64); }
TEST(matvec_i4_sdot4_mt_matches_w4a8_naive_g128) { check_matvec_i4_sdot_impl("sdot4_mt", 512, 1024, 128); }
TEST(matvec_i4_sdot4_mt_matches_w4a8_naive_g48) { check_matvec_i4_sdot_impl("sdot4_mt", 512, 544, 48); }

// ---------------------------------------------------------------------------
// sdot4 vs sdot3 逐位一致：唯一改动是组头转换的实现（硬件 FCVT 替软件
// 转换），两者都是无损的 fp16→fp32，输出必须逐位相同。
// ---------------------------------------------------------------------------
TEST(matvec_i4_sdot4_bitexact_vs_sdot3) {
    if (!tinyqwen::set_matvec_i4_impl_by_name("sdot3")) return;  // 平台无 dotprod
    const int out_dim = 128, in_dim = 896, group_size = 64;
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 5252);
    const std::vector<float> x = random_vec(in_dim, 5253);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> y3(out_dim), y4(out_dim);
    tinyqwen::set_matvec_i4_impl_by_name("sdot3");
    tinyqwen::matvec_i4(packed.data(), x.data(), y3.data(), out_dim, in_dim, group_size);
    tinyqwen::set_matvec_i4_impl_by_name("sdot4");
    tinyqwen::matvec_i4(packed.data(), x.data(), y4.data(), out_dim, in_dim, group_size);
    for (int i = 0; i < out_dim; ++i) {
        if (y3[i] != y4[i]) {  // 逐位比较（不是近似）
            TQ_FAIL("sdot4 vs sdot3 非逐位一致 @ row " + std::to_string(i) +
                    ": sdot3=" + std::to_string(y3[i]) + " sdot4=" + std::to_string(y4[i]));
            return;
        }
    }
    tinyqwen::set_matvec_i4_impl_by_name("ref");  // 复位
}

// ---------------------------------------------------------------------------
// 小矩阵走单线程分支也要对
// ---------------------------------------------------------------------------
TEST(matvec_i4_neon_mt_small_singlethread) { check_matvec_i4_impl("neon_mt", 32, 64, 64); }

// ---------------------------------------------------------------------------
// 测试：matmul_i4（prefill 路径）与 matvec_i4 的一致性
//
// matmul_i4 处理 N=2 列主序输入，每列的结果应等于对应 matvec_i4 的输出。
// 验证 prefill 批量化路径与 decode 单 token 路径数值等价。
// ---------------------------------------------------------------------------
TEST(matmul_i4_matches_matvec) {
    tinyqwen::set_matvec_i4_impl_by_name("ref");
    const int M = 48, K = 128, N = 2, G = 64;  // M=output dim, K=input dim, N=batch
    const std::vector<float> w = random_vec(static_cast<size_t>(M) * K, 99);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, M, K, G);
    const std::vector<float> x0 = random_vec(K, 1);  // 第 0 列输入
    const std::vector<float> x1 = random_vec(K, 2);  // 第 1 列输入
    // 构造列主序 [K, N] 输入矩阵
    std::vector<float> xKN(static_cast<size_t>(K) * N);
    for (int k = 0; k < K; ++k) { xKN[k] = x0[k]; xKN[K + k] = x1[k]; }
    // 执行 matmul
    std::vector<float> yMN(static_cast<size_t>(M) * N);
    tinyqwen::matmul_i4(packed.data(), xKN.data(), yMN.data(), M, K, N, G);
    // 分别执行两次 matvec 作为参照
    std::vector<float> y0(M), y1(M);
    tinyqwen::matvec_i4(packed.data(), x0.data(), y0.data(), M, K, G);
    tinyqwen::matvec_i4(packed.data(), x1.data(), y1.data(), M, K, G);
    // 逐元素对比：matmul 的第 0 列 == matvec(x0)，第 1 列 == matvec(x1)
    for (int m = 0; m < M; ++m) {
        EXPECT_NEAR(yMN[m], y0[m], 1e-4);       // 第 0 列
        EXPECT_NEAR(yMN[M + m], y1[m], 1e-4);   // 第 1 列
    }
}

// ---------------------------------------------------------------------------
// 测试：dispatch 未知实现名 fail fast
//
// 验证 set_matvec_i4_impl_by_name 对无效名称返回 false，
// 且不会破坏当前已注册的实现。
// ---------------------------------------------------------------------------
TEST(matvec_i4_dispatch_unknown_fails) {
    EXPECT_TRUE(!tinyqwen::set_matvec_i4_impl_by_name("no_such_impl"));  // 无效名 → false
    EXPECT_TRUE(tinyqwen::set_matvec_i4_impl_by_name("ref"));            // 有效名 → true
    EXPECT_EQ(std::string(tinyqwen::matvec_i4_impl_name()), std::string("ref"));  // 确认当前实现
}

// ---------------------------------------------------------------------------
// dequant_i4_to_f32（Qwen3.5 批量 prefill 的权重反量化）
//
// 独立对照：直接从 pack_i4_rtn 产出的 interleaved 字节逐组读 scale/zero +
// nibble，按 (q - zero) * scale 计算期望值，与 dequant 输出逐元素比对。
// 重点回归"组内偏移写到行内正确列"——曾出过每组都覆盖行首（out+col+i 写成
// out+i）导致整批 logits 错乱的 bug。
// ---------------------------------------------------------------------------
TEST(dequant_i4_to_f32_matches_naive) {
    const int out_dim = 40, in_dim = 192, group_size = 64;  // 192/64=3 组，整除
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 777);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> got(static_cast<size_t>(out_dim) * in_dim);
    tinyqwen::dequant_i4_to_f32(packed.data(), got.data(), out_dim, in_dim, group_size);
    // 逐组逐元素从字节独立还原期望值
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    double max_err = 0.0;
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            for (int i = start; i < end; ++i) {
                const int li = i - start;                       // 组内下标
                const uint8_t byte = gp[kHdr + li / 2];
                const int q = (li % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                const float expect = (static_cast<float>(q) - z) * s;
                const double err = std::fabs(static_cast<double>(got[o * in_dim + i]) - expect);
                max_err = std::max(max_err, err);
            }
        }
    }
    if (max_err >= 1e-5) {
        TQ_FAIL("dequant_i4_to_f32 max_err=" + std::to_string(max_err));
    }
}

// 尾部组（in_dim 非 group_size 整除）也要对——验证最后一组 elems 截断正确
TEST(dequant_i4_to_f32_partial_group) {
    const int out_dim = 24, in_dim = 200, group_size = 64;  // 200/64=3 余 8
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 778);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> got(static_cast<size_t>(out_dim) * in_dim);
    tinyqwen::dequant_i4_to_f32(packed.data(), got.data(), out_dim, in_dim, group_size);
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    double max_err = 0.0;
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            for (int i = start; i < end; ++i) {
                const int li = i - start;
                const uint8_t byte = gp[kHdr + li / 2];
                const int q = (li % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                const float expect = (static_cast<float>(q) - z) * s;
                const double err = std::fabs(static_cast<double>(got[o * in_dim + i]) - expect);
                max_err = std::max(max_err, err);
            }
        }
    }
    if (max_err >= 1e-5) {
        TQ_FAIL("dequant_i4_to_f32 partial max_err=" + std::to_string(max_err));
    }
}

// 大矩阵走线程池路径（>= 262144 元素）：验证并行反量化与朴素参考逐位一致
// （并行只改执行顺序，不改任何一组的数值）
TEST(dequant_i4_to_f32_parallel_pool_path) {
    const int out_dim = 512, in_dim = 896, group_size = 64;  // 458752 元素 > 阈值
    const std::vector<float> w = random_vec(static_cast<size_t>(out_dim) * in_dim, 779);
    const std::vector<uint8_t> packed = pack_i4_rtn(w, out_dim, in_dim, group_size);
    std::vector<float> got(static_cast<size_t>(out_dim) * in_dim);
    tinyqwen::dequant_i4_to_f32(packed.data(), got.data(), out_dim, in_dim, group_size);
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    const int group_total = kHdr + group_size / 2;
    const int row_bytes = groups_per_row * group_total;
    double max_err = 0.0;
    for (int o = 0; o < out_dim; ++o) {
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *gp = packed.data() + static_cast<size_t>(o) * row_bytes + g * group_total;
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, gp, 2);
            std::memcpy(&zero_h, gp + 2, 2);
            const float s = half_to_float(scale_h);
            const float z = half_to_float(zero_h);
            const int start = g * group_size;
            const int end = std::min(start + group_size, in_dim);
            for (int i = start; i < end; ++i) {
                const int li = i - start;
                const uint8_t byte = gp[kHdr + li / 2];
                const int q = (li % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                const float expect = (static_cast<float>(q) - z) * s;
                const double err = std::fabs(static_cast<double>(got[o * in_dim + i]) - expect);
                max_err = std::max(max_err, err);
            }
        }
    }
    if (max_err >= 1e-5) {
        TQ_FAIL("dequant_i4_to_f32 pool max_err=" + std::to_string(max_err));
    }
}
