// ============================================================================
// matvec_i4_ref.cpp — INT4 weight-only matvec 参考实现：y = W @ x
// ============================================================================
// 本文件是 INT4 量化权重矩阵乘向量的"标准答案"（对应 f32 族的 matvec_f32_ref）。
//
// 内存布局：非对称 uint4 [0,15]，per-group scale + zero_point（均 fp16），
// interleaved 布局。每组 [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(G/2 B)]。
// 反量化公式：float_val = (uint4_val - zero_point) × scale。
// 低 nibble 在前：byte & 0x0F = 偶数下标，byte >> 4 = 奇数下标。
//
// 数值约定：double 累加（最保守路径），所有 i4 优化变体必须与本实现对齐后
// 才能声称"算对了"。永不修改、永不删除。
//
// 注册名："ref"——dispatch 中 matvec_i4 的默认/兜底实现。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#include <cstring>    // std::memcpy（安全读取未对齐的 fp16）

namespace tinyqwen {

namespace {
// 每组头部字节数：scale(fp16, 2B) + zero_point(fp16, 2B) = 4B
constexpr int kGroupHeader = 4;
} // namespace

// ========================================================================
// matvec_i4_ref() — INT4 量化权重矩阵乘向量参考实现
// ========================================================================
// 功能：计算 y[out_dim] = W_i4[out_dim, in_dim] @ x[in_dim]
// 参数：
//   w          — INT4 packed 权重矩阵（interleaved 格式）
//   x          — fp32 输入向量（长度 in_dim）
//   y          — fp32 输出向量（长度 out_dim，调用方分配）
//   out_dim    — 输出维度（W 的行数）
//   in_dim     — 输入维度（W 的列数 = x 的长度）
//   group_size — 量化分组大小（典型值 32/64/128）
// 算法步骤：
//   1. 外层遍历 out_dim 行；
//   2. 中层遍历该行的所有量化组；
//   3. 内层逐元素解包 uint4 → 反量化 → double 乘累加。
// 返回值：无（结果写入 y）
void matvec_i4_ref(const uint8_t *w, const float *x, float *y,
                   int out_dim, int in_dim, int group_size) {
    // 计算每行有多少个量化组（向上取整）
    const int groups_per_row = (in_dim + group_size - 1) / group_size;
    // 每个组的 packed 数据字节数：group_size 个 uint4 挤成 group_size/2 字节
    const int group_data_bytes = group_size / 2;
    // 每个组的总字节数 = 头部(4B) + packed 数据
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    // 一整行权重的字节步长
    const int row_bytes = groups_per_row * group_total_bytes;

    // 外层循环：逐行处理
    for (int o = 0; o < out_dim; ++o) {
        // 定位当前行的起始地址
        const uint8_t *row = w + static_cast<size_t>(o) * row_bytes;
        // double 累加器：避免 float 连加的舍入误差累积
        double acc = 0.0;
        // col 跟踪当前处理到的列位置（用于尾组边界判断）
        int col = 0;

        // 中层循环：遍历该行的所有量化组
        for (int g = 0; g < groups_per_row; ++g) {
            // 定位当前组在行内的起始地址
            const uint8_t *group_ptr = row + g * group_total_bytes;

            // 用 memcpy 安全读取可能未对齐的 fp16 scale 和 zero_point
            uint16_t scale_h, zero_h;
            std::memcpy(&scale_h, group_ptr, 2);       // 前 2 字节是 scale（fp16）
            std::memcpy(&zero_h, group_ptr + 2, 2);    // 接下来 2 字节是 zero_point（fp16）
            // 将 fp16 转为 fp32 供后续计算使用
            const float scale = half_to_float(scale_h);
            const float zero = half_to_float(zero_h);

            // packed 数据紧跟在头部之后
            const uint8_t *packed = group_ptr + kGroupHeader;
            // 当前组的实际元素数：最后一组可能不足 group_size
            const int group_elems = (col + group_size <= in_dim) ? group_size : (in_dim - col);

            // 内层循环：逐元素解包 uint4 → 反量化 → double 乘累加
            for (int i = 0; i < group_elems; ++i) {
                const int byte_idx = i / 2; // 两个 uint4 共享一个字节
                uint8_t val;
                if (i % 2 == 0) {
                    val = packed[byte_idx] & 0x0F;       // 偶数下标：取低 nibble
                } else {
                    val = (packed[byte_idx] >> 4) & 0x0F; // 奇数下标：取高 nibble
                }
                // 反量化公式：float_val = (uint4_val - zero_point) × scale
                const float dequant = (static_cast<float>(val) - zero) * scale;
                // 提升为 double 后相乘并累加
                acc += static_cast<double>(dequant) * static_cast<double>(x[col + i]);
            }
            // 推进列位置到下一组
            col += group_size;
        }
        // 将 double 累加结果转回 float，写入输出
        y[o] = static_cast<float>(acc);
    }
}

// 自注册进 dispatch：matvec_i4 的 "ref" 实现（默认/兜底）
TINYQWEN_MATVEC_I4_VARIANT(matvec_i4_ref, "ref");

} // namespace tinyqwen
