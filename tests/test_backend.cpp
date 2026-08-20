// 后端抽象接口的单元测试

#include "test_framework.h"

#include <cmath>
#include <vector>

#include "backend.h"
#include "backend_cpu.h"

using namespace tinyqwen;

TEST (cpu_backend_create) {
    auto backend = create_cpu_backend();
    EXPECT_TRUE(backend != nullptr);
}

TEST (cpu_backend_rmsnorm) {
    auto backend = create_cpu_backend();
    const int n = 8;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> weight(n, 1.0f);
    std::vector<float> y(n);

    backend->rmsnorm(x.data(), weight.data(), y.data(), n, 1e-6f);

    // 简单检查：输出不为零
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += std::fabs(y[i]);
    EXPECT_TRUE(sum > 0.0f);
}

TEST (cpu_backend_argmax) {
    auto backend = create_cpu_backend();
    const int n = 5;
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

    int idx = backend->argmax(logits.data(), n);
    EXPECT_EQ(idx, 3);
}

TEST (cpu_backend_swiglu) {
    auto backend = create_cpu_backend();
    const int n = 4;
    std::vector<float> gate = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> up = {0.5f, 0.5f, 0.5f, 0.5f};

    backend->swiglu(gate.data(), up.data(), n);

    // swiglu(gate, up) = silu(gate) * up
    // silu(x) = x * sigmoid(x)
    // sigmoid(1) ≈ 0.731, silu(1) ≈ 0.731
    // 0.731 * 0.5 ≈ 0.366
    EXPECT_NEAR(gate[0], 0.366f, 0.01f);
}
