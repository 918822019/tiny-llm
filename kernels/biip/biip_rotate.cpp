// ============================================================================
// biip_rotate.cpp — BiIP 激活旋转（推理时对激活施加旋转的"逆"变换）
// ============================================================================
// 背景：KronQ 的 BiIP 旋转把权重变换为
//     W_rot = blockHadamard( (W * scaleH) ⊙ sign )   （沿输入维/列）
// 为使输出不变，推理时必须对激活施加配对变换：
//     x_rot = blockHadamard( (x / scaleH) ⊙ sign )
// 数学自抵消：y = x_rot @ W_rot^T = x @ W^T（Hadamard 正交 + scaleH 对消）。
//
// 本文件实现激活侧变换（标量参考实现，正确性锚点）：
//     1. y[j] = x[j] / scale[j]        （scale==nullptr 表示被中和，跳过）
//     2. y[j] *= sign[j]               （±1 符号翻转）
//     3. 每 block_size 个元素做归一化 Walsh-Hadamard（butterfly 快速变换）
//
// 归一化约定与 kronq/hadamard.py 的 get_hadamard_matrix（H/sqrt(size)）一致：
// butterfly 先算未归一化 ±1 Hadamard，最后整体乘 1/sqrt(block_size)。
// ============================================================================

#include "dispatch.h"

#include <cmath>    // std::sqrt

namespace tinyqwen {

// ========================================================================
// biip_rotate_activation() — 对激活向量施加 BiIP 配对旋转
// ========================================================================
// 参数：
//   x          — 输入激活（长度 dim，fp32）
//   y          — 输出激活（长度 dim，fp32；可与 x 同址，原地安全）
//   dim        — 向量长度（= 该子层 in_features，须被 block_size 整除）
//   scale      — scaleH 向量 [dim]（fp32）；nullptr = 被中和，跳过除法
//   sign       — ±1 符号向量 [dim]（fp32）
//   block_size — Hadamard 分块大小（2 的幂）
// 说明：butterfly 每块做 log2(block_size) 级加减，最后乘 1/sqrt(block_size)。
void biip_rotate_activation(const float *x, float *y, int dim,
                            const float *scale, const float *sign, int block_size) {
    // 1+2. 逐元素：除 scale、乘 sign
    if (scale) {
        for (int j = 0; j < dim; ++j) {
            y[j] = (x[j] / scale[j]) * sign[j];
        }
    } else {
        for (int j = 0; j < dim; ++j) {
            y[j] = x[j] * sign[j];
        }
    }

    // 3. 分块归一化 Walsh-Hadamard（butterfly）
    const int num_blocks = dim / block_size;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(block_size));
    for (int b = 0; b < num_blocks; ++b) {
        float *blk = y + static_cast<size_t>(b) * block_size;
        for (int h = 1; h < block_size; h <<= 1) {
            for (int i = 0; i < block_size; i += 2 * h) {
                for (int j = i; j < i + h; ++j) {
                    const float a = blk[j];
                    const float c = blk[j + h];
                    blk[j] = a + c;
                    blk[j + h] = a - c;
                }
            }
        }
        for (int j = 0; j < block_size; ++j) {
            blk[j] *= inv_sqrt;
        }
    }
}

} // namespace tinyqwen
