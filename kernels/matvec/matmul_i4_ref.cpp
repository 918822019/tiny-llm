// ============================================================================
// matmul_i4_ref.cpp — INT4 量化权重的矩阵乘法参考实现（Y = W_i4 × X）
// ============================================================================
// 本文件实现 INT4 weight-only 量化的通用矩阵乘法（matmul），是 matvec_i4_ref
// 的"多列推广版"：matvec 只算 y = W @ x（x 是单列向量），而 matmul 同时
// 处理 N 列输入 X[K,N]，输出 Y[M,N]。在 LLM 推理中主要用于 prefill 阶段
// （一次处理整个 prompt 的多 token）或 lm_head 投影。
//
// 内存布局：
//   - W_i4：interleaved INT4 packed 格式（与 matvec_i4_ref 相同），每行 M 维、
//     K 个元素按 group_size 分组，每组 [scale_fp16(2B) | zero_fp16(2B) |
//     packed_uint4(group_size/2 B)]；低 nibble 存偶数下标、高 nibble 存奇数下标。
//   - X：列主序 [K, N]，每列是一个 token 的输入向量（连续 K 个 float）。
//   - Y：列主序 [M, N]，每列是对应 token 的输出向量。
//
// 算法：对 N 列各做一次完整的 M×K matvec，等价于循环调用 matvec_i4_ref。
// 写成独立 kernel 是为了后续做 tiling / 分块优化时不必改调用侧逻辑。
//
// 数值约定：double 累加（最保守路径），所有 i4 matmul 优化变体必须与本实现
// 对齐后才算"算对了"。
//
// 注册名："ref"——dispatch 中 matmul_i4 的默认/兜底实现。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATMUL_I4_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float 等辅助函数

#include <cstring>    // std::memcpy（安全读取未对齐的 fp16 scale/zero）

namespace tinyqwen {
namespace {

// 每组头部的字节数：scale(fp16, 2B) + zero_point(fp16, 2B) = 4B
constexpr int kGroupHeader = 4;

// =========================================================================
// matmul_i4_ref() — INT4 量化权重矩阵乘法参考实现
// =========================================================================
// 功能：计算 Y[M,N] = W_i4[M,K] × X[K,N]
// 参数：
//   w          — INT4 packed 权重矩阵（interleaved 格式，大小由 M/K/group_size 决定）
//   x          — 输入矩阵，列主序 [K, N]
//   y          — 输出矩阵，列主序 [M, N]（调用方分配）
//   M          — 输出行数（权重矩阵的行数）
//   K          — 收缩维度（权重矩阵的列数 = 输入矩阵的行数）
//   N          — 输出列数（一次处理的 token 数）
//   group_size — 量化分组大小（典型值 32/64/128，必须是 2 的倍数）
// 算法步骤：
//   1. 外层遍历 N 列，每列是一次独立的 matvec；
//   2. 中层遍历 M 行，每行与当前列做点积；
//   3. 内层遍历该行的所有 group，逐组解包 uint4 → 反量化 → 乘累加。
// 返回值：无（结果写入 y）
void matmul_i4_ref(const uint8_t *w, const float *x, float *y,
                   int M, int K, int N, int group_size) {
    // 计算每行有多少个量化组（向上取整，最后一组可能不满 group_size 个元素）
    const int groups_per_row = (K + group_size - 1) / group_size;
    // 每个组的 packed 数据字节数：group_size 个 uint4 挤成 group_size/2 字节
    const int group_data_bytes = group_size / 2;
    // 每个组的总字节数 = 头部(4B) + packed 数据
    const int group_total_bytes = kGroupHeader + group_data_bytes;
    // 一整行权重的字节步长 = 组数 × 每组字节数
    const int row_bytes = groups_per_row * group_total_bytes;

    // 外层循环：逐列处理（每列是一个 token 的输入向量）
    for (int col = 0; col < N; ++col) {
        // 定位当前列在 X 中的起始地址（列主序：第 col 列从 col*K 开始）
        const float *xc = x + static_cast<size_t>(col) * K;
        // 定位当前列在 Y 中的起始地址（列主序：第 col 列从 col*M 开始）
        float *yc = y + static_cast<size_t>(col) * M;

        // 中层循环：逐行处理（每行算出当前列输出的一个分量）
        for (int row = 0; row < M; ++row) {
            // 定位当前行在 W 中的起始地址
            const uint8_t *rp = w + static_cast<size_t>(row) * row_bytes;
            // double 累加器：避免 float 连加的舍入误差累积
            double acc = 0.0;
            // k 跟踪当前处理到的列位置（用于尾组边界判断）
            int k = 0;

            // 内层循环：遍历该行的所有量化组
            for (int g = 0; g < groups_per_row; ++g) {
                // 定位当前组在行内的起始地址
                const uint8_t *gp = rp + g * group_total_bytes;
                // 用 memcpy 安全读取可能未对齐的 fp16 scale 和 zero_point
                uint16_t scale_h, zero_h;
                std::memcpy(&scale_h, gp, 2);       // 前 2 字节是 scale（fp16）
                std::memcpy(&zero_h, gp + 2, 2);    // 接下来 2 字节是 zero_point（fp16）
                // 将 fp16 转为 fp32 供后续计算使用
                const float scale = half_to_float(scale_h);
                const float zero = half_to_float(zero_h);

                // packed 数据紧跟在头部之后
                const uint8_t *packed = gp + kGroupHeader;
                // 当前组的实际元素数：最后一组可能不足 group_size（K 不是 group_size 倍数时）
                const int elems = (k + group_size <= K) ? group_size : (K - k);

                // 逐元素解包 uint4 → 反量化 → 乘累加
                for (int i = 0; i < elems; ++i) {
                    // 解包 uint4：低 nibble（& 0x0F）存偶数下标，高 nibble（>> 4）存奇数下标
                    uint8_t val = (i % 2 == 0) ? (packed[i / 2] & 0x0F)
                                               : ((packed[i / 2] >> 4) & 0x0F);
                    // 反量化公式：float_val = (uint4_val - zero_point) × scale
                    float dequant = (static_cast<float>(val) - zero) * scale;
                    // 乘上对应的输入元素并累加到 double 累加器
                    acc += static_cast<double>(dequant) * static_cast<double>(xc[k + i]);
                }
                // 推进列位置到下一组
                k += group_size;
            }
            // 将 double 累加结果转回 float，写入输出的对应位置
            yc[row] = static_cast<float>(acc);
        }
    }
}

} // namespace
// 自注册进 dispatch：matmul_i4 的 "ref" 实现（默认/兜底）
TINYQWEN_MATMUL_I4_VARIANT(matmul_i4_ref, "ref");
} // namespace tinyqwen
