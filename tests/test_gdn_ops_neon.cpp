// GDN 算子 NEON 变体的正确性门禁。
//
// 与 test_ops_neon.cpp 同款约定：set_ops_impl_by_name("neon") 失败就 [skip]。
// 数值锚点：每个 NEON kernel vs 对应 _ref 在相同输入上对比，按容差门禁。

#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dispatch.h"
#include "gdn_ops.h"

using namespace tinyqwen;

namespace {
    struct Lcg {
        uint32_t s = 54321u;
        float next(float lo, float hi) {
            s = s * 1664525u + 1013904223u;
            const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
            return lo + u * (hi - lo);
        }
    };

    float max_abs_diff(const float *a, const float *b, int n) {
        float mx = 0.0f;
        for (int i = 0; i < n; ++i) {
            float d = std::fabs(a[i] - b[i]);
            if (d > mx) mx = d;
        }
        return mx;
    }
} // namespace

TEST (l2norm_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {128, 127, 64, 5};
    Lcg rng;
    for (int n : sizes) {
        std::vector<float> x_ref(n), x_neon(n);
        for (int i = 0; i < n; ++i) {
            x_ref[i] = x_neon[i] = rng.next(-2.0f, 2.0f);
        }
        l2norm_inplace_ref(x_ref.data(), n, 1e-6f);
        l2norm_inplace(x_neon.data(), n, 1e-6f); // dispatch -> neon
        float err = max_abs_diff(x_ref.data(), x_neon.data(), n);
        if (err > 1e-5f) {
            std::printf("l2norm n=%d max_err=%e\n", n, err);
        }
        EXPECT_TRUE(err <= 1e-5f);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (rmsnorm_gated_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {128, 127, 64};
    Lcg rng;
    for (int n : sizes) {
        std::vector<float> x(n), gate(n), weight(n), y_ref(n), y_neon(n);
        for (int i = 0; i < n; ++i) {
            x[i] = rng.next(-1.0f, 1.0f);
            gate[i] = rng.next(-3.0f, 3.0f);
            weight[i] = rng.next(0.5f, 1.5f);
        }
        rmsnorm_gated_ref(x.data(), gate.data(), weight.data(), y_ref.data(), n, 1e-6f);
        rmsnorm_gated(x.data(), gate.data(), weight.data(), y_neon.data(), n, 1e-6f);
        float err = max_abs_diff(y_ref.data(), y_neon.data(), n);
        if (err > 1e-4f) {
            std::printf("rmsnorm_gated n=%d max_err=%e\n", n, err);
        }
        EXPECT_TRUE(err <= 1e-4f);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (conv1d_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // dim=6144（真实）和 dim=13（含尾段，非 4 整除）。
    const int dims[] = {6144, 13};
    const int K = 4;
    const int state_len = K - 1;
    Lcg rng;
    for (int dim : dims) {
        std::vector<float> state_ref(dim * state_len), state_neon(dim * state_len);
        std::vector<float> weight(dim * K);
        std::vector<float> x(dim), out_ref(dim), out_neon(dim);
        for (int i = 0; i < dim * state_len; ++i)
            state_ref[i] = state_neon[i] = rng.next(-1.0f, 1.0f);
        for (int i = 0; i < dim * K; ++i)
            weight[i] = rng.next(-0.5f, 0.5f);
        for (int i = 0; i < dim; ++i)
            x[i] = rng.next(-1.0f, 1.0f);

        causal_conv1d_update_ref(x.data(), state_ref.data(), weight.data(),
                                 out_ref.data(), dim, K);
        causal_conv1d_update(x.data(), state_neon.data(), weight.data(),
                             out_neon.data(), dim, K);

        float out_err = max_abs_diff(out_ref.data(), out_neon.data(), dim);
        float state_err = max_abs_diff(state_ref.data(), state_neon.data(), dim * state_len);
        if (out_err > 1e-5f || state_err > 1e-6f) {
            std::printf("conv1d dim=%d out_err=%e state_err=%e\n", dim, out_err, state_err);
        }
        EXPECT_TRUE(out_err <= 1e-5f);
        EXPECT_TRUE(state_err <= 1e-6f);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (conv1d_neon_inplace) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 就地测试：x 和 out 指向同一缓冲区。
    const int dim = 6144, K = 4, state_len = K - 1;
    Lcg rng;
    std::vector<float> state_ref(dim * state_len), state_neon(dim * state_len);
    std::vector<float> weight(dim * K);
    std::vector<float> buf_ref(dim), buf_neon(dim);
    for (int i = 0; i < dim * state_len; ++i)
        state_ref[i] = state_neon[i] = rng.next(-1.0f, 1.0f);
    for (int i = 0; i < dim * K; ++i)
        weight[i] = rng.next(-0.5f, 0.5f);
    for (int i = 0; i < dim; ++i)
        buf_ref[i] = buf_neon[i] = rng.next(-1.0f, 1.0f);

    // in-place: out == x
    causal_conv1d_update_ref(buf_ref.data(), state_ref.data(), weight.data(),
                             buf_ref.data(), dim, K);
    causal_conv1d_update(buf_neon.data(), state_neon.data(), weight.data(),
                         buf_neon.data(), dim, K);

    float out_err = max_abs_diff(buf_ref.data(), buf_neon.data(), dim);
    float state_err = max_abs_diff(state_ref.data(), state_neon.data(), dim * state_len);
    EXPECT_TRUE(out_err <= 1e-5f);
    EXPECT_TRUE(state_err <= 1e-6f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (gdn_step_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int qk_dim = 128, v_dim = 128;
    const int state_size = qk_dim * v_dim;
    Lcg rng;

    std::vector<float> S_ref(state_size), S_neon(state_size);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    for (int i = 0; i < state_size; ++i)
        S_ref[i] = S_neon[i] = rng.next(-0.1f, 0.1f);
    for (int i = 0; i < qk_dim; ++i) {
        q[i] = rng.next(-1.0f, 1.0f);
        k[i] = rng.next(-1.0f, 1.0f);
    }
    for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);

    const float g = -0.5f;
    const float beta = 0.7f;

    gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), g, beta,
                 o_ref.data(), qk_dim, v_dim);
    gdn_step(S_neon.data(), q.data(), k.data(), v.data(), g, beta,
             o_neon.data(), qk_dim, v_dim);

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    if (o_err > 1e-4f || s_err > 1e-5f) {
        std::printf("gdn_step o_err=%e s_err=%e\n", o_err, s_err);
    }
    EXPECT_TRUE(o_err <= 1e-4f);
    EXPECT_TRUE(s_err <= 1e-5f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (gdn_step_neon_multi_step) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 10 步连续递归，验证漂移不累积到不可接受。
    const int qk_dim = 128, v_dim = 128;
    const int state_size = qk_dim * v_dim;
    Lcg rng;

    std::vector<float> S_ref(state_size, 0.0f), S_neon(state_size, 0.0f);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    for (int step = 0; step < 10; ++step) {
        for (int i = 0; i < qk_dim; ++i) {
            q[i] = rng.next(-1.0f, 1.0f);
            k[i] = rng.next(-1.0f, 1.0f);
        }
        for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);
        const float g = rng.next(-1.0f, -0.01f);
        const float beta = rng.next(0.1f, 0.9f);

        gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), g, beta,
                     o_ref.data(), qk_dim, v_dim);
        gdn_step(S_neon.data(), q.data(), k.data(), v.data(), g, beta,
                 o_neon.data(), qk_dim, v_dim);
    }

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    if (o_err > 5e-4f || s_err > 1e-4f) {
        std::printf("gdn_step 10-step drift: o_err=%e s_err=%e\n", o_err, s_err);
    }
    EXPECT_TRUE(o_err <= 5e-4f);
    EXPECT_TRUE(s_err <= 1e-4f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (gdn_step_neon_small_dims) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 非 4 整除的小维度（fake model 用 qk_dim=8, v_dim=8）。
    const int qk_dim = 8, v_dim = 8;
    const int state_size = qk_dim * v_dim;
    Lcg rng;

    std::vector<float> S_ref(state_size), S_neon(state_size);
    std::vector<float> q(qk_dim), k(qk_dim), v(v_dim);
    std::vector<float> o_ref(v_dim), o_neon(v_dim);

    for (int i = 0; i < state_size; ++i)
        S_ref[i] = S_neon[i] = rng.next(-0.1f, 0.1f);
    for (int i = 0; i < qk_dim; ++i) {
        q[i] = rng.next(-1.0f, 1.0f);
        k[i] = rng.next(-1.0f, 1.0f);
    }
    for (int i = 0; i < v_dim; ++i) v[i] = rng.next(-1.0f, 1.0f);

    gdn_step_ref(S_ref.data(), q.data(), k.data(), v.data(), -0.3f, 0.5f,
                 o_ref.data(), qk_dim, v_dim);
    gdn_step(S_neon.data(), q.data(), k.data(), v.data(), -0.3f, 0.5f,
             o_neon.data(), qk_dim, v_dim);

    float o_err = max_abs_diff(o_ref.data(), o_neon.data(), v_dim);
    float s_err = max_abs_diff(S_ref.data(), S_neon.data(), state_size);
    EXPECT_TRUE(o_err <= 1e-4f);
    EXPECT_TRUE(s_err <= 1e-5f);
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}
