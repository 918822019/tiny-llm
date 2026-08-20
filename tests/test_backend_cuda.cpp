// CUDA 后端的单元测试（仅 CUDA 构建编译）

#include "test_framework.h"

#ifdef TINYQWEN_HAS_CUDA

#include <cmath>
#include <vector>

#include "backend.h"
#include "backend_cuda.h"

using namespace tinyqwen;

TEST (cuda_backend_create) {
    auto backend = create_cuda_backend();
    EXPECT_TRUE(backend != nullptr);
}

TEST (cuda_backend_rmsnorm) {
    auto backend = create_cuda_backend();
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

TEST (cuda_backend_argmax) {
    auto backend = create_cuda_backend();
    const int n = 5;
    std::vector<float> logits = {1.0f, 3.0f, 2.0f, 5.0f, 4.0f};

    int idx = backend->argmax(logits.data(), n);
    EXPECT_EQ(idx, 3);
}

TEST (cuda_backend_swiglu) {
    auto backend = create_cuda_backend();
    const int n = 4;
    std::vector<float> gate = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> up = {0.5f, 0.5f, 0.5f, 0.5f};

    backend->swiglu(gate.data(), up.data(), n);

    // swiglu(gate, up) = silu(gate) * up
    // sigmoid(1) ≈ 0.731, silu(1) ≈ 0.731
    // 0.731 * 0.5 ≈ 0.366
    EXPECT_NEAR(gate[0], 0.366f, 0.01f);
}

TEST (cuda_backend_matvec_f32) {
    auto backend = create_cuda_backend();
    const int out_dim = 4;
    const int in_dim = 8;

    // 创建简单的测试权重（identity-like）
    std::vector<float> w(out_dim * in_dim);
    for (int i = 0; i < out_dim; ++i) {
        for (int j = 0; j < in_dim; ++j) {
            w[i * in_dim + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> y(out_dim);

    WeightTensor wt{w.data(), QuantType::kF32, out_dim, in_dim, 0};
    backend->matvec(wt, x.data(), y.data(), out_dim, in_dim);

    // y = W @ x，W 是 identity-like，所以 y[i] ≈ x[i]
    for (int i = 0; i < out_dim; ++i) {
        EXPECT_NEAR(y[i], x[i], 1e-4f);
    }
}

#endif // TINYQWEN_HAS_CUDA
