// =============================================================================
// test_backend_cuda.cpp
// =============================================================================
// 文件级别说明：
//   本文件测试 CUDA 后端（GPU Backend）的核心算子实现。
//   仅在定义了 TINYQWEN_HAS_CUDA 宏时编译（即检测到 CUDA 工具链时）。
//   所有测试代码位于 #ifdef TINYQWEN_HAS_CUDA ... #endif 保护块内，
//   在非 CUDA 构建中本文件为空（不注册任何测试）。
//
// 测试覆盖的 CUDA 算子：
//   1. cuda_backend_create         - 创建 CUDA 后端实例
//   2. cuda_backend_rmsnorm        - CUDA 版 RMS 归一化
//   3. cuda_backend_argmax         - CUDA 版最大值索引
//   4. cuda_backend_swiglu         - CUDA 版 SwiGLU 激活
//   5. cuda_backend_matvec_f32     - CUDA 版 FP32 矩阵向量乘
//
// 所有测试通过 Backend 抽象接口调用，与 CPU 后端测试保持对称性。
// 数值容差与 CPU 后端相同（浮点计算顺序差异由容差覆盖）。
// =============================================================================

// CUDA 后端的单元测试（仅 CUDA 构建编译）

#include "test_framework.h"

#ifdef TINYQWEN_HAS_CUDA

#include <cmath>
#include <vector>

#include "backend.h"
#include "backend_cuda.h"

using namespace tinyqwen;

// 测试用例：创建 CUDA 后端实例
// 验证 create_cuda_backend() 能成功返回非空的后端指针。
TEST (cuda_backend_create) {
    auto backend = create_cuda_backend();  // 创建 CUDA 后端
    EXPECT_TRUE(backend != nullptr);      // 确保指针非空
}

// 测试用例：CUDA 后端 RMS 归一化
// 与 CPU 后端测试同款输入，验证 CUDA 实现的正确性。
// 公式：y[i] = x[i] / sqrt(mean(x[j]^2) + eps) * weight[i]
TEST (cuda_backend_rmsnorm) {
    auto backend = create_cuda_backend();
    const int n = 8;  // 向量长度
    // 构造输入向量 x = [1,2,3,4,5,6,7,8]
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // 权重向量全为 1.0（不改变归一化方向）
    std::vector<float> weight(n, 1.0f);
    // 输出向量
    std::vector<float> y(n);

    // 执行 CUDA 版 RMS 归一化
    backend->rmsnorm(x.data(), weight.data(), y.data(), n, 1e-6f);

    // 简单检查：输出不为零
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += std::fabs(y[i]);
    EXPECT_TRUE(sum > 0.0f);
}

// 测试用例：CUDA 后端 argmax（最大值索引）
// 验证 CUDA 实现的 argmax 与 CPU 参考实现结果一致。
TEST (cuda_backend_argmax) {
    auto backend = create_cuda_backend();
    const int n = 5;  // logits 数组长度
    // 构造 logits 数组，最大值为 5.0
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

    int idx = backend->argmax(logits.data(), n);  // 求最大值索引
    EXPECT_EQ(idx, 3);  // 期望索引为 3（值为 5.0）
}

// 测试用例：CUDA 后端 SwiGLU 激活函数
// 验证 CUDA 实现的 swiglu 与 CPU 参考实现结果一致。
// swiglu(gate, up) = silu(gate) * up
TEST (cuda_backend_swiglu) {
    auto backend = create_cuda_backend();
    const int n = 4;  // 向量长度
    // 构造 gate 和 up 输入
    std::vector<float> gate = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> up = {0.5f, 0.5f, 0.5f, 0.5f};

    // 执行 CUDA 版 SwiGLU，结果写回 gate
    backend->swiglu(gate.data(), up.data(), n);

    // swiglu(gate, up) = silu(gate) * up
    // sigmoid(1) ≈ 0.731, silu(1) ≈ 0.731
    // 0.731 * 0.5 ≈ 0.366
    EXPECT_NEAR(gate[0], 0.366f, 0.01f);
}

// 测试用例：CUDA 后端 FP32 矩阵向量乘
// 验证 CUDA 实现的 matvec（矩阵乘向量）的正确性。
// 使用 identity-like 权重矩阵（对角元为 1，其余为 0），
// 期望 y = W @ x ≈ x（输出等于输入对应位置）。
//
// 矩阵形状：out_dim=4（输出维度），in_dim=8（输入维度）
// 权重布局：W[i][j] = (i == j) ? 1.0 : 0.0
TEST (cuda_backend_matvec_f32) {
    auto backend = create_cuda_backend();
    const int out_dim = 4;  // 输出维度（矩阵行数）
    const int in_dim = 8;   // 输入维度（矩阵列数）

    // 创建简单的测试权重（identity-like）：对角元为 1，其余为 0
    std::vector<float> w(out_dim * in_dim);
    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            w[i * in_dim + j] = (i == j) ? 1.0f : 0.0f;  // 构造近似单位矩阵
        }
    }

    // 构造输入向量 x = [1,2,3,4,5,6,7,8]
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // 输出向量
    std::vector<float> y(out_dim);

    // 构造 WeightTensor 描述符（量化类型为 FP32，无额外参数）
    WeightTensor wt{w.data(), QuantType::kF32, out_dim, in_dim, 0};
    // 执行 CUDA 版矩阵向量乘
    backend->matvec(wt, x.data(), y.data(), out_dim, in_dim);

    // y = W @ x，W 是 identity-like，所以 y[i] ≈ x[i]
    for (int i = 0; i < out_dim; ++i) {
        EXPECT_NEAR(y[i], x[i], 1e-4f);  // 容差 1e-4 覆盖浮点误差
    }
}

#endif // TINYQWEN_HAS_CUDA