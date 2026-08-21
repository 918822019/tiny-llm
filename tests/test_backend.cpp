// =============================================================================
// test_backend.cpp
// =============================================================================
// 文件级别说明：
//   本文件测试 CPU 后端（Backend 抽象接口）的核心算子实现。
//   后端抽象层（Backend）是 tinyqwen 的计算抽象，允许在不同硬件平台
//   （CPU、CUDA GPU）上切换计算实现。CPU 后端是默认的参考实现。
//
// 测试覆盖的算子：
//   1. create_cpu_backend         - 创建 CPU 后端实例
//   2. cpu_backend_rmsnorm        - RMS 归一化（RMS Normalization）
//   3. cpu_backend_argmax         - 求最大值索引（贪婪采样用）
//   4. cpu_backend_swiglu         - SiLU 门控激活函数（SwiGLU）
//
// 所有测试通过 Backend 抽象接口调用，而非直接调用底层函数，
// 以确保接口层的正确性。
// =============================================================================

// 后端抽象接口的单元测试

#include "test_framework.h"

#include <cmath>
#include <vector>

#include "backend.h"
#include "backend_cpu.h"

using namespace tinyqwen;

// 测试用例：创建 CPU 后端实例
// 验证 create_cpu_backend() 能成功返回非空的后端指针。
TEST (cpu_backend_create) {
    auto backend = create_cpu_backend();  // 创建 CPU 后端
    EXPECT_TRUE(backend != nullptr);      // 确保指针非空
}

// 测试用例：CPU 后端 RMS 归一化
// 验证 rmsnorm() 方法对长度为 8 的向量执行 RMS 归一化后，
// 输出向量的绝对值之和大于 0（即非全零输出）。
// 公式：y[i] = x[i] / sqrt(mean(x[j]^2) + eps) * weight[i]
TEST (cpu_backend_rmsnorm) {
    auto backend = create_cpu_backend();
    const int n = 8;  // 向量长度
    // 构造输入向量 x = [1,2,3,4,5,6,7,8]
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // 权重向量全为 1.0（不改变归一化方向）
    std::vector<float> weight(n, 1.0f);
    // 输出向量（将被填充）
    std::vector<float> y(n);

    // 执行 RMS 归一化，eps = 1e-6 防止除零
    backend->rmsnorm(x.data(), weight.data(), y.data(), n, 1e-6f);

    // 简单检查：输出不为零（如果全零说明计算有误）
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += std::fabs(y[i]);
    EXPECT_TRUE(sum > 0.0f);
}

// 测试用例：CPU 后端 argmax（最大值索引）
// 验证 argmax() 方法在给定 logits 数组中找到第一个最大值的索引。
// 输入 [1,3,2,5,4] 的最大值是 5，位于索引 3。
TEST (cpu_backend_argmax) {
    auto backend = create_cpu_backend();
    const int n = 5;  // logits 数组长度
    // 构造 logits 数组，最大值为 5.0
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

    int idx = backend->argmax(logits.data(), n);  // 求最大值索引
    EXPECT_EQ(idx, 3);  // 期望索引为 3（值为 5.0）
}

// 测试用例：CPU 后端 SwiGLU 激活函数
// 验证 swiglu() 方法对 gate 和 up 输入执行 SwiGLU 操作：
//   swiglu(gate, up) = silu(gate) * up
//   其中 silu(x) = x * sigmoid(x)
//   结果写回 gate 数组（就地修改）。
//
// 对于输入 gate=1.0, up=0.5：
//   sigmoid(1) ≈ 0.731, silu(1) ≈ 0.731
//   swiglu = 0.731 * 0.5 ≈ 0.366
TEST (cpu_backend_swiglu) {
    auto backend = create_cpu_backend();
    const int n = 4;  // 向量长度
    // 构造 gate 和 up 输入
    std::vector<float> gate = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> up = {0.5f, 0.5f, 0.5f, 0.5f};

    // 执行 SwiGLU，结果写回 gate
    backend->swiglu(gate.data(), up.data(), n);

    // swiglu(gate, up) = silu(gate) * up
    // silu(x) = x * sigmoid(x)
    // sigmoid(1) ≈ 0.731, silu(1) ≈ 0.731
    // 0.731 * 0.5 ≈ 0.366
    EXPECT_NEAR(gate[0], 0.366f, 0.01f);  // 容差 0.01 覆盖浮点误差
}