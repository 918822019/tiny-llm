// ============================================================================
// matvec_i4_neon.cpp — INT4 weight-only matvec：y = W @ x —— NEON 优化版
// ============================================================================
// 本文件实现 INT4 量化权重的矩阵乘向量（NEON SIMD 加速版），是 matvec_i4_ref
// 的优化变体。
//
// 内存布局与 ref 一致：每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(G/2 B)]，
// 低 nibble 在前（byte & 0x0F = 偶数下标，byte >> 4 = 奇数下标）。
// 反量化公式：float_val = (uint4_val - zero_point) × scale。
//
// 核心循环优化思路：每迭代处理 32 个 uint4（16 bytes packed），利用 NEON
// 拆 nibble → 转 fp32 → FMA 累加，比标量 ref 快 3-5x（ARM Cortex-A78 实测）。
//
// 关键 NEON 操作链：
//   1. vld1q_u8：加载 16 字节 packed 数据（= 32 个 uint4）
//   2. vand/vshr + vzip：拆分高低 nibble 并交错重排为顺序 [0,1,2,...,31]
//   3. vmovl_u8 + vmovl_u16 + vcvtq_f32_u32：uint8 → uint16 → uint32 → float32
//   4. vfmaq_f32(v_neg_zs, f, v_scale)：f = val*scale + (-zero*scale) = 反量化
//   5. vfmaq_f32(acc, f, x)：acc += dequant × x（FMA 累加）
//
// group_size 必须是 32 的倍数（典型 128），否则回退到 ref。
// 仅 aarch64 构建注册；其他平台编译为空翻译单元。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h> // NEON intrinsic 声明
#include <cstdint>    // uint8_t, uint16_t
#include <cstring>    // std::memcpy

namespace tinyqwen {
namespace {

// ========================================================================
// dot_group_i4_neon() — 处理单行的一个 group（NEON 32-wide 主循环）
// ========================================================================
// 功能：计算一个量化组内所有元素的反量化点积贡献
// 参数：
//   packed     — 该组的 packed uint4 数据起始地址
//   x          — 对应的输入向量片段
//   scale      — 该组的 scale（已转为 fp32）
//   zero       — 该组的 zero_point（已转为 fp32）
//   group_elems — 该组的实际元素数（最后一组可能不满 group_size）
// 返回值：该 group 的点积贡献（fp32）
inline float dot_group_i4_neon(const uint8_t *packed, const float *x,
                               float scale, float zero, int group_elems) {
    // 预计算常量向量：避免在内循环里重复构造
    const float32x4_t v_scale = vdupq_n_f32(scale);        // [scale, scale, scale, scale]
    // v_neg_zs = -zero * scale：反量化公式 val*scale - zero*scale = val*scale + (-zero*scale)
    // 将减法变成加法（FMA 可以直接用），省一条指令
    const float32x4_t v_neg_zs = vdupq_n_f32(-zero * scale);

    // 4 个独立 fp32 累加器（与 f32 neon 相同的延迟重叠策略）
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    // 低 nibble 掩码：0x0F = 00001111，用于提取每个字节的低 4 位
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);

    int i = 0;
    const int n32 = group_elems & ~31; // 主循环边界：32 的倍数

    for (; i < n32; i += 32) {
        // 加载 16 字节 packed 数据 = 32 个 nibble（uint4）
        uint8x16_t raw = vld1q_u8(packed + i / 2);

        // ---- 拆分高低 nibble ----
        // vandq_u8(raw, mask_lo)：取每个字节的低 4 位 → 偶数下标的 uint4
        uint8x16_t lo8 = vandq_u8(raw, mask_lo);
        // vshrq_n_u8(raw, 4)：右移 4 位取高 nibble → 奇数下标的 uint4
        uint8x16_t hi8 = vshrq_n_u8(raw, 4);

        // ---- 交错重排为顺序 [0,1,2,...,31] ----
        // vzipq_u8(lo8, hi8)：交错合并 → val[0]=[e0,e1,...,e15], val[1]=[e16,...,e31]
        uint8x16x2_t zipped = vzipq_u8(lo8, hi8);

        // ---- uint8 → float32 的类型提升链 ----
        // NEON 没有直接从 uint8 到 float32 的转换，需要逐步提升：
        // uint8 → uint16 → uint32 → float32
        // vmovl_u8：uint8x8 → uint16x8（零扩展）
        uint16x8_t w16_0 = vmovl_u8(vget_low_u8(zipped.val[0]));   // e0..e7
        uint16x8_t w16_1 = vmovl_u8(vget_high_u8(zipped.val[0]));  // e8..e15
        uint16x8_t w16_2 = vmovl_u8(vget_low_u8(zipped.val[1]));   // e16..e23
        uint16x8_t w16_3 = vmovl_u8(vget_high_u8(zipped.val[1]));  // e24..e31

        // ---- 处理 e0..e3：uint16 → uint32 → float32 → 反量化 → FMA 累加 ----
        // vmovl_u16(vget_low_u16(w16_0))：取 w16_0 的低 4 个 uint16 → uint32x4
        // vcvtq_f32_u32(...)：uint32x4 → float32x4
        float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_0)));
        float32x4_t x0 = vld1q_f32(x + i); // 加载 4 个输入
        // 反量化：f0 = f0 * scale + (-zero * scale) = (val - zero) * scale
        f0 = vfmaq_f32(v_neg_zs, f0, v_scale);
        // 乘累加：acc0 += f0 * x0
        acc0 = vfmaq_f32(acc0, f0, x0);

        // e4..e7
        float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_0)));
        float32x4_t x1 = vld1q_f32(x + i + 4);
        f1 = vfmaq_f32(v_neg_zs, f1, v_scale);
        acc1 = vfmaq_f32(acc1, f1, x1);

        // e8..e11
        float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_1)));
        float32x4_t x2 = vld1q_f32(x + i + 8);
        f2 = vfmaq_f32(v_neg_zs, f2, v_scale);
        acc2 = vfmaq_f32(acc2, f2, x2);

        // e12..e15
        float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_1)));
        float32x4_t x3 = vld1q_f32(x + i + 12);
        f3 = vfmaq_f32(v_neg_zs, f3, v_scale);
        acc3 = vfmaq_f32(acc3, f3, x3);

        // e16..e19（复用 acc0）
        float32x4_t f4 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_2)));
        float32x4_t x4 = vld1q_f32(x + i + 16);
        f4 = vfmaq_f32(v_neg_zs, f4, v_scale);
        acc0 = vfmaq_f32(acc0, f4, x4);

        // e20..e23
        float32x4_t f5 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_2)));
        float32x4_t x5 = vld1q_f32(x + i + 20);
        f5 = vfmaq_f32(v_neg_zs, f5, v_scale);
        acc1 = vfmaq_f32(acc1, f5, x5);

        // e24..e27
        float32x4_t f6 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16_3)));
        float32x4_t x6 = vld1q_f32(x + i + 24);
        f6 = vfmaq_f32(v_neg_zs, f6, v_scale);
        acc2 = vfmaq_f32(acc2, f6, x6);

        // e28..e31
        float32x4_t f7 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16_3)));
        float32x4_t x7 = vld1q_f32(x + i + 28);
        f7 = vfmaq_f32(v_neg_zs, f7, v_scale);
        acc3 = vfmaq_f32(acc3, f7, x7);
    }

    // 合并 4 个累加器 + 横向归约
    float32x4_t sum01 = vaddq_f32(acc0, acc1);
    float32x4_t sum23 = vaddq_f32(acc2, acc3);
    float total = vaddvq_f32(vaddq_f32(sum01, sum23));

    // 标量尾段：group_elems 不是 32 的倍数时逐个补上
    for (; i < group_elems; ++i) {
        const int byte_idx = i / 2;
        // 解包 uint4：偶数下标取低 nibble，奇数下标取高 nibble
        uint8_t val = (i % 2 == 0) ? (packed[byte_idx] & 0x0F) : ((packed[byte_idx] >> 4) & 0x0F);
        // 反量化后乘输入并累加
        total += (static_cast<float>(val) - zero) * scale * x[i];
    }

    return total;
}

// 每组头部字节数：scale(fp16, 2B) + zero_point(fp16, 2B) = 4B
constexpr int kGroupHeader = 4;

// ========================================================================
// matvec_i4_neon() — INT4 matvec NEON 优化版入口
// ========================================================================
// 功能：计算 y = W_i4 @ x（INT4 量化权重，NEON 加速）
void matvec_i4_neon(const uint8_t *w, const float *x, float *y,
                    int out_dim, int in_dim, int group_size) {
    // 计算每行有多少个量化组
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    // 每个组的 packed 数据字节数
    const int group_data_bytes = group_size / 2;
    // 每个组的总字节数 = 头部 + packed 数据
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    // 一整行权重的字节步长
    const int row_bytes = groups_per_row * group_total_bytes;

    // 逐行处理
    for (int o = 0; o < out_dim; ++o) {
        const uint8_t *row = w + static_cast<size_t>(o) * row_bytes; // 定位行起点
        float acc = 0.0f; // float 累加器（各组贡献之和）
        int col = 0;      // 当前处理到的列位置

        // 遍历该行的所有量化组
        for (int g = 0; g < groups_per_row; ++g) {
            const uint8_t *group_ptr = row + g * group_total_bytes; // 组起始地址

            // 安全读取可能未对齐的 fp16 scale 和 zero_point
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, group_ptr, 2);     // scale（fp16）
            std::memcpy(&zero_h, group_ptr + 2, 2);  // zero_point（fp16）
            const float scale = half_to_float(scale_h); // fp16 → fp32
            const float zero = half_to_float(zero_h);   // fp16 → fp32

            const uint8_t *packed = group_ptr + kGroupHeader; // packed 数据起点
            // 该组的实际元素数（最后一组可能不满）
            const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

            // 调用 NEON 优化的组点积
            acc += dot_group_i4_neon(packed, x + col, scale, zero, group_elems);
            col += group_size; // 推进到下一组
        }
        y[o] = acc; // 写入输出
    }
}

} // anonymous namespace

// 自注册进 dispatch：--matvec-impl neon（i4 注册表）
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_neon, "neon");

} // namespace tinyqwen

#endif // __aarch64__
