// 非 matvec 算子 NEON 变体的正确性门禁。
//
// 约定与 test_matvec.cpp 的 neon 门禁同款：neon 只在 aarch64 构建注册，
// set_ops_impl_by_name("neon") 失败就打印 [skip] 返回（非静默假通过——先确认
// 实现存在才比较）。数值锚点：argmax 与 ref 逐位一致（EXPECT_EQ 下标）；
// rmsnorm/rope/attention/swiglu 是 ulp/多项式逼近级差异，用容差门禁。
// swiglu_ref 另测"与原两步逐位一致"（融合等价性）。

#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include "dispatch.h"
#include "ref_ops.h"

using namespace tinyqwen;

namespace {
    // 简单确定性伪随机（可复现，不依赖 <random> 的分布实现细节）。
    struct Lcg {
        uint32_t s = 12345u;
        float next(float lo, float hi) {
            s = s * 1664525u + 1013904223u;
            const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);
            return lo + u * (hi - lo);
        }
    };
} // namespace

TEST (ops_dispatch_selects_impl) {
    // "neon" 仅 aarch64 注册。先探测；没有就整体 skip（其余 neon 用例同理）。
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    EXPECT_TRUE(std::string(ops_impl_name()) == "neon");
    // 恢复到 ref（兜底默认）。
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
    EXPECT_TRUE(std::string(ops_impl_name()) == "ref");
    // 未知名字拒绝。
    EXPECT_TRUE(!set_ops_impl_by_name("no_such_ops"));
    EXPECT_TRUE(std::string(ops_impl_name()) == "ref");
}

TEST (argmax_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 多个尺寸（含尾段）+ 刻意制造平局。argmax 是精确 max + 首个相等，
    // 应与 ref 逐位一致 -> 断言下标相等（不是容差）。
    const int sizes[] = {1, 3, 4, 7, 1001, 151936};
    for (int n : sizes) {
        std::vector<float> logits(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) logits[i] = rng.next(-10.0f, 10.0f);
        // 制造平局：让若干位置并列最大值，验证"取靠前者"。
        if (n >= 8) {
            const float big = 42.0f;
            logits[n / 3] = big;
            logits[n / 2] = big;
            logits[n / 4] = big; // 最靠前的平局位置
        }
        const int expected = argmax_ref(logits.data(), n);
        const int got = argmax(logits.data(), n); // dispatch -> neon
        EXPECT_EQ(got, expected);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (rmsnorm_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 896（真实 hidden，16 的倍数）+ 897（尾段）+ 1。float 累加平方和 vs
    // ref double 累加 -> ulp 级，容差 1e-5 充分宽裕。
    const int sizes[] = {1, 897, 896};
    for (int n : sizes) {
        std::vector<float> x(n), w(n), y_ref(n), y_neon(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) {
            x[i] = rng.next(-3.0f, 3.0f);
            w[i] = rng.next(0.5f, 2.0f);
        }
        rmsnorm_ref(x.data(), w.data(), y_ref.data(), n, 1e-6f);
        rmsnorm(x.data(), w.data(), y_neon.data(), n, 1e-6f); // dispatch -> neon
        for (int i = 0; i < n; ++i) EXPECT_NEAR(y_neon[i], y_ref[i], 1e-5);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (rope_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // GQA：14 q heads / 2 kv heads，head_dim=64（half=32，4 的倍数 + 无尾段）。
    const int n_heads = 14, n_kv = 2, head_dim = 64;
    const int qn = n_heads * head_dim, kn = n_kv * head_dim;
    std::vector<float> q0(qn), k0(kn);
    Lcg rng;
    for (int i = 0; i < qn; ++i) q0[i] = rng.next(-2.0f, 2.0f);
    for (int i = 0; i < kn; ++i) k0[i] = rng.next(-2.0f, 2.0f);

    const float theta = 1e6f;
    for (int pos : {0, 1, 5, 35}) {
        std::vector<float> qr = q0, kr = k0, qn_ = q0, kn_ = k0;
        rope_ref(qr.data(), kr.data(), n_heads, n_kv, head_dim, pos, theta);
        rope(qn_.data(), kn_.data(), n_heads, n_kv, head_dim, pos, theta); // neon
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(qn_[i], qr[i], 1e-5);
        for (int i = 0; i < kn; ++i) EXPECT_NEAR(kn_[i], kr[i], 1e-5);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (attention_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // GQA + strided KV cache（max_seq_len > seq_len，验证跨步）。
    const int n_heads = 14, n_kv = 2, head_dim = 64;
    const int max_seq_len = 40;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const size_t stride = static_cast<size_t>(max_seq_len) * head_dim;

    std::vector<float> q(n_heads * head_dim);
    std::vector<float> kcache(n_kv * stride), vcache(n_kv * stride);
    Lcg rng;
    for (auto &v : q) v = rng.next(-1.0f, 1.0f);
    for (auto &v : kcache) v = rng.next(-1.0f, 1.0f);
    for (auto &v : vcache) v = rng.next(-1.0f, 1.0f);

    for (int seq_len : {1, 5, 35}) {
        std::vector<float> out_ref(n_heads * head_dim), out_neon(n_heads * head_dim);
        attention_decode_ref(q.data(), kcache.data(), vcache.data(), seq_len, max_seq_len,
                             n_heads, n_kv, head_dim, scale, out_ref.data());
        attention_decode(q.data(), kcache.data(), vcache.data(), seq_len, max_seq_len, n_heads,
                         n_kv, head_dim, scale, out_neon.data()); // neon
        for (size_t i = 0; i < out_ref.size(); ++i) {
            EXPECT_NEAR(out_neon[i], out_ref[i], 1e-4);
        }
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

TEST (swiglu_ref_equals_two_step) {
    // 融合等价性锚点：swiglu_ref 必须与原来的"silu_ref + 逐元素乘 up"逐位一致。
    const int n = 4865; // 含尾段
    std::vector<float> gate(n), up(n);
    Lcg rng;
    for (int i = 0; i < n; ++i) {
        gate[i] = rng.next(-6.0f, 6.0f);
        up[i] = rng.next(-3.0f, 3.0f);
    }
    std::vector<float> two_step = gate;
    silu_ref(two_step.data(), two_step.data(), n);
    for (int i = 0; i < n; ++i) two_step[i] *= up[i];

    std::vector<float> fused = gate;
    swiglu_ref(fused.data(), up.data(), n);
    for (int i = 0; i < n; ++i) EXPECT_TRUE(fused[i] == two_step[i]);
}

TEST (swiglu_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 常规区间 + 大 |x|（溢出边界，验证 exp 钳位语义与 ref 一致趋于 0/x）。
    const int sizes[] = {4864, 4865};
    for (int n : sizes) {
        std::vector<float> gate(n), up(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) {
            gate[i] = rng.next(-8.0f, 8.0f);
            up[i] = rng.next(-3.0f, 3.0f);
        }
        // 注入极端值。
        gate[0] = 100.0f;
        gate[1] = -100.0f;
        gate[2] = 0.0f;

        std::vector<float> ref = gate;
        swiglu_ref(ref.data(), up.data(), n);
        std::vector<float> neon = gate;
        swiglu(neon.data(), up.data(), n); // dispatch -> neon
        for (int i = 0; i < n; ++i) {
            // exp 多项式逼近 ~1e-7 相对；放宽到 1e-4 覆盖极端值的绝对误差。
            EXPECT_NEAR(neon[i], ref[i], 1e-4 + 1e-4 * std::fabs(ref[i]));
        }
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}
