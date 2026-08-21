// ============================================================================
// test_ops_neon.cpp — 非 matvec 算子 NEON 变体的正确性门禁
// ============================================================================
// 本文件验证 ARM NEON SIMD 优化的通用算子与参考实现（_ref）的数值一致性。
// 覆盖的算子：
//   - argmax          — 取最大值的索引
//   - rmsnorm         — RMSNorm 归一化
//   - rope            — 旋转位置编码
//   - attention_decode — Decode 阶段的注意力计算
//   - swiglu          — SwiGLU 激活（含融合等价性验证）
//
// 约定（与 test_matvec.cpp 的 neon 门禁同款）：
//   - neon 只在 aarch64 构建注册，set_ops_impl_by_name("neon") 失败就
//     打印 [skip] 返回（非静默假通过——先确认实现存在才比较）
//   - argmax 与 ref 逐位一致（EXPECT_EQ 下标，精确匹配）
//   - rmsnorm/rope/attention/swiglu 是 ulp/多项式逼近级差异，用容差门禁
//   - swiglu_ref 另测"与原两步逐位一致"（融合等价性锚点）
// ============================================================================

#include "test_framework.h"  // 自研测试框架

#include <cmath>      // std::fabs, std::sqrt
#include <cstdint>    // uint32_t
#include <vector>     // std::vector

#include "dispatch.h"  // set_ops_impl_by_name / ops_impl_name（算子 dispatch）
#include "ref_ops.h"   // *_ref 参考实现声明

using namespace tinyqwen;

namespace {
    // =========================================================================
    // struct Lcg — 简单确定性伪随机数生成器（Linear Congruential Generator）
    // =========================================================================
    // 可复现，不依赖 <random> 的分布实现细节。
    // 参数来自 Numerical Recipes：s = s * 1664525 + 1013904223
    struct Lcg {
        uint32_t s = 12345u;  // 固定种子
        float next(float lo, float hi) {
            s = s * 1664525u + 1013904223u;  // LCG 步进
            const float u = static_cast<float>(s >> 8) / static_cast<float>(1u << 24);  // [0,1)
            return lo + u * (hi - lo);       // 映射到 [lo, hi)
        }
    };
} // namespace

// ---------------------------------------------------------------------------
// 测试：ops dispatch 机制本身的选择行为
//
// 验证 set_ops_impl_by_name 能正确切换实现、拒绝无效名称、查询当前实现名。
// ---------------------------------------------------------------------------
TEST (ops_dispatch_selects_impl) {
    // "neon" 仅 aarch64 注册。先探测；没有就整体 skip
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 确认切换到 neon
    EXPECT_TRUE(std::string(ops_impl_name()) == "neon");
    // 恢复到 ref（兜底默认）
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
    EXPECT_TRUE(std::string(ops_impl_name()) == "ref");
    // 未知名字应被拒绝，且不影响当前实现
    EXPECT_TRUE(!set_ops_impl_by_name("no_such_ops"));
    EXPECT_TRUE(std::string(ops_impl_name()) == "ref");
}

// ---------------------------------------------------------------------------
// 测试：argmax NEON vs ref
//
// 多个尺寸（含尾段）+ 刻意制造平局。argmax 是精确 max + 首个相等，
// 应与 ref 逐位一致 → 断言下标相等（不是容差）。
// ---------------------------------------------------------------------------
TEST (argmax_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    // 多种尺寸：1(极小), 3(小), 4(SIMD对齐), 7(尾段), 1001(大), 151936(真实词表)
    const int sizes[] = {1, 3, 4, 7, 1001, 151936};
    for (int n : sizes) {
        std::vector<float> logits(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) logits[i] = rng.next(-10.0f, 10.0f);
        // 制造平局：让若干位置并列最大值，验证"取靠前者"
        if (n >= 8) {
            const float big = 42.0f;
            logits[n / 3] = big;
            logits[n / 2] = big;
            logits[n / 4] = big;  // 最靠前的平局位置
        }
        const int expected = argmax_ref(logits.data(), n);  // ref 结果
        const int got = argmax(logits.data(), n);            // dispatch -> neon
        EXPECT_EQ(got, expected);  // 精确匹配（argmax 无浮点误差）
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：rmsnorm NEON vs ref
//
// 896（真实 hidden，16 的倍数）+ 897（尾段）+ 1（极小）。
// float 累加平方和 vs ref double 累加 → ulp 级差异，容差 1e-5 充分宽裕。
// ---------------------------------------------------------------------------
TEST (rmsnorm_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {1, 897, 896};
    for (int n : sizes) {
        std::vector<float> x(n), w(n), y_ref(n), y_neon(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) {
            x[i] = rng.next(-3.0f, 3.0f);    // 输入
            w[i] = rng.next(0.5f, 2.0f);     // 缩放权重
        }
        rmsnorm_ref(x.data(), w.data(), y_ref.data(), n, 1e-6f);
        rmsnorm(x.data(), w.data(), y_neon.data(), n, 1e-6f);  // dispatch -> neon
        for (int i = 0; i < n; ++i) EXPECT_NEAR(y_neon[i], y_ref[i], 1e-5);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：rope NEON vs ref
//
// GQA：14 q heads / 2 kv heads，head_dim=64（half=32，4 的倍数 + 无尾段）。
// theta=1e6（大基数，减少三角函数精度问题的影响）。
// 在 pos={0,1,5,35} 下验证。
// ---------------------------------------------------------------------------
TEST (rope_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int n_heads = 14, n_kv = 2, head_dim = 64;
    const int qn = n_heads * head_dim, kn = n_kv * head_dim;  // Q/K 总维度
    std::vector<float> q0(qn), k0(kn);
    Lcg rng;
    for (int i = 0; i < qn; ++i) q0[i] = rng.next(-2.0f, 2.0f);
    for (int i = 0; i < kn; ++i) k0[i] = rng.next(-2.0f, 2.0f);

    const float theta = 1e6f;  // RoPE 基频
    for (int pos : {0, 1, 5, 35}) {
        // 复制输入（rope 就地修改）
        std::vector<float> qr = q0, kr = k0, qn_ = q0, kn_ = k0;
        rope_ref(qr.data(), kr.data(), n_heads, n_kv, head_dim, pos, theta);
        rope(qn_.data(), kn_.data(), n_heads, n_kv, head_dim, pos, theta);  // neon
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(qn_[i], qr[i], 1e-5);
        for (int i = 0; i < kn; ++i) EXPECT_NEAR(kn_[i], kr[i], 1e-5);
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：attention_decode NEON vs ref
//
// GQA + strided KV cache（max_seq_len > seq_len，验证跨步读取）。
// n_heads=14, n_kv=2, head_dim=64, max_seq_len=40。
// 在 seq_len={1,5,35} 下验证。
// 容差 1e-4：attention 涉及 softmax + 加权求和，累积误差较大。
// ---------------------------------------------------------------------------
TEST (attention_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int n_heads = 14, n_kv = 2, head_dim = 64;
    const int max_seq_len = 40;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));  // 1/sqrt(d_k)
    const size_t stride = static_cast<size_t>(max_seq_len) * head_dim;   // KV cache 跨步

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
                         n_kv, head_dim, scale, out_neon.data());  // neon
        for (size_t i = 0; i < out_ref.size(); ++i) {
            EXPECT_NEAR(out_neon[i], out_ref[i], 1e-4);
        }
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}

// ---------------------------------------------------------------------------
// 测试：swiglu_ref 融合等价性锚点
//
// 验证 swiglu_ref（融合版本）必须与原来的"silu_ref + 逐元素乘 up"逐位一致。
// 这是后续 NEON swiglu 对比的前提——如果 ref 自身不等价，NEON 对标就失去意义。
// ---------------------------------------------------------------------------
TEST (swiglu_ref_equals_two_step) {
    const int n = 4865;  // 含尾段（非 SIMD 对齐）
    std::vector<float> gate(n), up(n);
    Lcg rng;
    for (int i = 0; i < n; ++i) {
        gate[i] = rng.next(-6.0f, 6.0f);
        up[i] = rng.next(-3.0f, 3.0f);
    }
    // 两步法：先 silu 再逐元素乘
    std::vector<float> two_step = gate;
    silu_ref(two_step.data(), two_step.data(), n);  // silu 就地
    for (int i = 0; i < n; ++i) two_step[i] *= up[i];  // 逐元素乘

    // 融合法
    std::vector<float> fused = gate;
    swiglu_ref(fused.data(), up.data(), n);

    // 逐位精确比较（同一实现，应完全相同）
    for (int i = 0; i < n; ++i) EXPECT_TRUE(fused[i] == two_step[i]);
}

// ---------------------------------------------------------------------------
// 测试：swiglu NEON vs ref
//
// 常规区间 + 大 |x|（溢出边界），验证 exp 钳位语义与 ref 一致。
// 注入极端值 gate[0]=100, gate[1]=-100, gate[2]=0 测试边界行为。
// 容差：1e-4 + 1e-4*|ref|（相对+绝对混合容差，覆盖极端值）。
// ---------------------------------------------------------------------------
TEST (swiglu_neon_matches_ref) {
    if (!set_ops_impl_by_name("neon")) {
        std::printf("[skip] current build has no 'neon' ops (not aarch64)\n");
        return;
    }
    const int sizes[] = {4864, 4865};  // 对齐 + 尾段
    for (int n : sizes) {
        std::vector<float> gate(n), up(n);
        Lcg rng;
        for (int i = 0; i < n; ++i) {
            gate[i] = rng.next(-8.0f, 8.0f);
            up[i] = rng.next(-3.0f, 3.0f);
        }
        // 注入极端值测试边界
        gate[0] = 100.0f;    // silu(100) ≈ 100（exp 溢出钳位）
        gate[1] = -100.0f;   // silu(-100) ≈ 0
        gate[2] = 0.0f;      // silu(0) = 0

        std::vector<float> ref = gate;
        swiglu_ref(ref.data(), up.data(), n);
        std::vector<float> neon = gate;
        swiglu(neon.data(), up.data(), n);  // dispatch -> neon
        for (int i = 0; i < n; ++i) {
            // exp 多项式逼近 ~1e-7 相对误差；放宽到 1e-4 覆盖极端值的绝对误差
            EXPECT_NEAR(neon[i], ref[i], 1e-4 + 1e-4 * std::fabs(ref[i]));
        }
    }
    EXPECT_TRUE(set_ops_impl_by_name("ref"));
}
