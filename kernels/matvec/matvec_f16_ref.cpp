// ============================================================================
// matvec_f16_ref.cpp — 矩阵乘向量（f16 权重参考实现）：y = W @ x
// ============================================================================
// 本文件是 f16 族的"标准答案"（对应 f32 族的 matvec_f32_ref）。
//
// 核心特征：
//   - 权重是 IEEE binary16（按 uint16_t 搬运，见 ref_ops.h 的 half_to_float），
//     x/y 仍是 fp32——weight-only 半精度只省权重的存储/搬运，计算精度不降；
//   - double 累加（最保守的数值路径），任何 f16 优化版先和它对齐才算"算对了"；
//   - 永不修改、永不删除，出问题时的兜底；
//   - 所有平台可编译（无 SIMD 依赖）——即使 aarch64 优化变体不存在，
//     f16 模型也永远有一条能跑的正确路径。
//
// 自注册进 f16 注册表的 "ref"：main 里 f16 模型的默认/兜底实现即本文件。
// ============================================================================

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT 自注册宏
#include "ref_ops.h"  // half_to_float：将 uint16_t 表示的 IEEE fp16 转为 float

namespace tinyqwen {
    // ========================================================================
    // matvec_f16_ref() — f16 权重矩阵乘向量参考实现
    // ========================================================================
    // 功能：计算 y[out_dim] = W[out_dim, in_dim] @ x[in_dim]
    // 参数：
    //   w        — f16 权重矩阵（以 uint16_t* 存储，行主序 [out_dim, in_dim]）
    //   x        — fp32 输入向量（长度 in_dim）
    //   y        — fp32 输出向量（长度 out_dim，调用方分配）
    //   out_dim  — 输出维度（W 的行数）
    //   in_dim   — 输入维度（W 的列数 = x 的长度）
    // 算法：逐行做点积，每行的 f16 权重先转 fp32 再用 double 累加。
    // 返回值：无（结果写入 y）
    void matvec_f16_ref(const uint16_t *w, const float *x, float *y, int out_dim,
                        int in_dim) {
        // 外层循环：逐行处理，每行算出 y 的一个分量
        for (int o = 0; o < out_dim; ++o) {
            // 定位第 o 行的起点（行主序：跳过前面 o 行，每行 in_dim 个 uint16_t）
            const uint16_t *row = w + static_cast<size_t>(o) * in_dim;
            // double 累加器：避免 float 连加的舍入误差累积
            double acc = 0.0;
            // 内层循环：逐元素做 f16→f32 转换 + 乘累加
            for (int i = 0; i < in_dim; ++i) {
                // half_to_float(row[i])：将 uint16_t 表示的 IEEE fp16 转为 fp32
                // 然后提升为 double 与 x[i]（隐式提升为 double）相乘并累加
                acc += static_cast<double>(half_to_float(row[i])) * x[i];
            }
            // 将 double 累加结果转回 float，写入输出的第 o 个分量
            y[o] = static_cast<float>(acc);
        }
    }

    // f16 注册表的 "ref"——与 f32 注册表的 "ref" 同名不同表，按模型 dtype 解析。
    // 这一行永远不删：它是 f16 模型的最终兜底。
    TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_ref, "ref");
} // namespace tinyqwen
