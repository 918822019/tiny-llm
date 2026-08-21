// ============================================================================
// matmul_ref.cpp — fp32 矩阵乘法参考实现（Y = W × X）
// ============================================================================
// 本文件实现标准的 fp32 矩阵乘法，是 matvec_f32_ref 的"多列推广版"：
// matvec 只算 y = W @ x（x 是单列向量），而 matmul 同时处理 N 列输入 X[K,N]，
// 输出 Y[M,N]。在 LLM 推理中主要用于 prefill 阶段（一次处理整个 prompt
// 的多 token）或 lm_head 投影。
//
// 内存布局：
//   - W：行主序 [M, K]，从 HuggingFace 导出时不做转置。同一行的 K 个元素
//     在内存中连续，因此每行点积都是顺序访存。
//   - X：列主序 [K, N]，每列是一个 token 的输入向量（连续 K 个 float）。
//   - Y：列主序 [M, N]，每列是对应 token 的输出向量。
//
// 算法：对 N 列各做一次完整的 M×K matvec，等价于循环调用 matvec_f32_ref。
// 写成独立 kernel 是为了后续做 tiling / 分块优化时不必改调用侧逻辑。
//
// 数值约定：double 累加（最保守路径），所有 fp32 matmul 优化变体必须与本
// 实现对齐后才算"算对了"。
//
// 注册名："ref"——dispatch 中 matmul 的默认/兜底实现。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATMUL_VARIANT 自注册宏
#include "ref_ops.h"  // 辅助函数声明

#include <cstring>    // 标准库（本文件暂未使用，但保留以备后续扩展）

namespace tinyqwen {
namespace {

// =========================================================================
// matmul_ref() — fp32 矩阵乘法参考实现
// =========================================================================
// 功能：计算 Y[M,N] = W[M,K] × X[K,N]
// 参数：
//   w — 权重矩阵，行主序 [M, K]
//   x — 输入矩阵，列主序 [K, N]
//   y — 输出矩阵，列主序 [M, N]（调用方分配）
//   M — 输出行数（权重矩阵的行数）
//   K — 收缩维度（权重矩阵的列数 = 输入矩阵的行数）
//   N — 输出列数（一次处理的 token 数）
// 算法步骤：
//   1. 外层遍历 N 列，每列是一次独立的 matvec；
//   2. 中层遍历 M 行，每行与当前列做点积；
//   3. 内层遍历 K 维，逐元素乘累加（double 精度）。
// 返回值：无（结果写入 y）
void matmul_ref(const float *w, const float *x, float *y, int M, int K, int N) {
    // 外层循环：逐列处理（每列是一个 token 的输入向量）
    for (int col = 0; col < N; ++col) {
        // 定位当前列在 X 中的起始地址（列主序：第 col 列从 col*K 开始）
        const float *xc = x + static_cast<size_t>(col) * K;
        // 定位当前列在 Y 中的起始地址（列主序：第 col 列从 col*M 开始）
        float *yc = y + static_cast<size_t>(col) * M;
        // 中层循环：逐行处理（每行算出当前列输出的一个分量）
        for (int row = 0; row < M; ++row) {
            // double 累加器：避免 float 连加的舍入误差累积
            double acc = 0.0;
            // 定位当前行在 W 中的起始地址（行主序：第 row 行从 row*K 开始）
            const float *wr = w + static_cast<size_t>(row) * K;
            // 内层循环：逐元素乘累加（点积的核心）
            for (int k = 0; k < K; ++k) {
                // 权重和输入都提升为 double 后相乘，避免 float 乘法的精度损失
                acc += static_cast<double>(wr[k]) * static_cast<double>(xc[k]);
            }
            // 将 double 累加结果转回 float，写入输出的对应位置
            yc[row] = static_cast<float>(acc);
        }
    }
}

} // namespace
// 自注册进 dispatch：matmul 的 "ref" 实现（默认/兜底）
TINYQWEN_MATMUL_VARIANT(matmul_ref, "ref");
} // namespace tinyqwen
