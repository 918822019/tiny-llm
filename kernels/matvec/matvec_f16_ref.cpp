// 矩阵乘向量（f16 权重参考实现）：y = W @ x。
//
// f16 族的"标准答案"（对应 f32 族的 matvec_f32_ref）：权重是 IEEE binary16
// （按 uint16_t 搬运，见 ref_ops.h 的 half_to_float），x/y 仍是 fp32——
// weight-only 半精度只省权重的存储/搬运，计算精度不降。
//
// 纪律与 f32 族完全相同：
//   - double 累加（最保守的数值路径），任何 f16 优化版先和它对齐才算"算对了"；
//   - 永不修改、永不删除，出问题时的兜底；
//   - 所有平台可编译（无 SIMD 依赖）——即使 aarch64 优化变体不存在，
//     f16 模型也永远有一条能跑的正确路径。
//
// 自注册进 f16 注册表的 "ref"：main 里 f16 模型的默认/兜底实现即本文件。

#include "dispatch.h" // TINYQWEN_MATVEC_F16_VARIANT 自注册宏
#include "ref_ops.h"

namespace tinyqwen {
    void matvec_f16_ref(const uint16_t *w, const float *x, float *y, int out_dim,
                        int in_dim) {
        for (int o = 0; o < out_dim; ++o) {
            const uint16_t *row = w + static_cast<size_t>(o) * in_dim;
            double acc = 0.0;
            for (int i = 0; i < in_dim; ++i) {
                acc += static_cast<double>(half_to_float(row[i])) * x[i];
            }
            y[o] = static_cast<float>(acc);
        }
    }

    // f16 注册表的 "ref"——与 f32 注册表的 "ref" 同名不同表，按模型 dtype 解析。
    TINYQWEN_MATVEC_F16_VARIANT(matvec_f16_ref, "ref");
} // namespace tinyqwen
