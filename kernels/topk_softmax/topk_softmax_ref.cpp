// ============================================================================
// topk_softmax_ref.cpp — MoE 路由门 top-k + softmax 参考实现（标量版）
// ============================================================================
// 本文件实现 MoE 路由的核心选择 + 归一算子：从路由门 logits 中选 k 个最大
// 值并对它们做 softmax 归一。
//
// 数学定义：
//   1. 从 gate_logits[n] 选出前 k 大（降序），记下标 indices[k]
//   2. 对选中的 k 个 logits 做数值稳定 softmax：
//        m = max(selected logits)
//        weights[i] = exp(logits[indices[i]] - m) / Σ_j exp(logits[indices[j]] - m)
//   结果 weights 和为 1，与 HF Qwen3 MoE 的 topk + softmax 路由一致。
//
// 平局约定：
//   分数并列时按靠前者优先（用稳定选择 + 严格 '>' 比较），保证 SIMD 重写后
//   数值/下标可复现。
//
// 注册名："ref"——dispatch 中 topk_softmax 的默认/兜底实现。仅 ops 分发
// （--ops-impl neon 时才会被覆盖；目前无 neon 变体，恒走本实现）。
// ============================================================================

#include "dispatch.h" // TINYQWEN_TOPK_SOFTMAX_VARIANT 自注册宏
#include "ref_ops.h"  // topk_softmax_ref 声明

#include <algorithm>  // std::partial_sort
#include <cmath>      // std::exp
#include <numeric>    // std::iota
#include <vector>

namespace tinyqwen {

// ========================================================================
// topk_softmax_ref — top-k 选择 + softmax 归一参考实现
// ========================================================================
// 参数：
//   gate_logits — 路由门输出 [n_experts]
//   n          — 专家数
//   k          — 激活专家数（top-k）
//   indices    — 输出，选中的专家下标 [k]（降序）
//   weights    — 输出，归一化路由权重 [k]（和为 1）
void topk_softmax_ref(const float *gate_logits, int n, int k,
                      int *indices, float *weights) {
    if (k <= 0 || n <= 0) return;
    const int kk = k > n ? n : k;

    // 取前 kk 大：用 partial_sort 按分数降序，稳定保留靠前者
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                      [gate_logits](int a, int b) {
                          // 严格 '>'：平局取靠前（partial_sort 对等价元素
                          // 不保证顺序，故用 a<b 时 a 在前以保证稳定）
                          return gate_logits[a] > gate_logits[b];
                      });

    // 数值稳定 softmax：先减选中集最大值
    float m = gate_logits[idx[0]];
    for (int i = 1; i < kk; ++i) {
        if (gate_logits[idx[i]] > m) m = gate_logits[idx[i]];
    }
    double sum = 0.0;
    for (int i = 0; i < kk; ++i) {
        const float e = std::exp(static_cast<double>(gate_logits[idx[i]]) - m);
        weights[i] = e;
        sum += static_cast<double>(e);
    }
    const double inv = (sum > 0.0) ? 1.0 / sum : 0.0;
    for (int i = 0; i < kk; ++i) {
        indices[i] = idx[i];
        weights[i] = static_cast<float>(static_cast<double>(weights[i]) * inv);
    }
    // 若 k > n，剩余槽位补 0（调用方不应依赖）
    for (int i = kk; i < k; ++i) {
        indices[i] = -1;
        weights[i] = 0.0f;
    }
}

// 自注册进 dispatch：topk_softmax 的 "ref" 实现（默认/兜底）
TINYQWEN_TOPK_SOFTMAX_VARIANT(topk_softmax_ref, "ref");

} // namespace tinyqwen
