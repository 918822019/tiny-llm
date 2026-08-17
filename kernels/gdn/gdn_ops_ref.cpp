// GDN reference kernels：标量、fp32、正确性优先。
// 每个函数的数学口径见 kernels/gdn_ops.h 的注释（与 HF torch 兜底实现对齐）。

#include "gdn_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace tinyqwen {
    namespace {
        inline float silu_scalar(float x) { return x / (1.0f + std::exp(-x)); }
    } // namespace

    void l2norm_inplace_ref(float *x, int n, float eps) {
        double sumsq = 0.0;
        for (int i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            sumsq += xi * xi;
        }
        const float inv_norm = 1.0f / std::sqrt(static_cast<float>(sumsq) + eps);
        for (int i = 0; i < n; ++i) x[i] *= inv_norm;
    }

    void causal_conv1d_update_ref(const float *x, float *conv_state, const float *weight,
                                  float *out, int dim, int kernel_size) {
        if (kernel_size < 2) {
            std::fprintf(stderr, "causal_conv1d_update_ref: kernel_size %d < 2\n", kernel_size);
            std::abort();
        }
        const int state_len = kernel_size - 1;
        for (int c = 0; c < dim; ++c) {
            float *st = conv_state + static_cast<size_t>(c) * state_len;
            const float *w = weight + static_cast<size_t>(c) * kernel_size;
            // 注意 x 与 out 可能同址（就地）：先把当前输入存下来，避免下面
            // out[c]=silu(...) 覆盖 x[c] 后，再往 state 里推进的是被覆盖的值。
            const float xc = x[c];
            // 卷积：state 里是最近 state_len 个输入（旧->新），拼上当前 xc。
            float acc = 0.0f;
            for (int i = 0; i < state_len; ++i) acc += w[i] * st[i];
            acc += w[state_len] * xc;
            out[c] = silu_scalar(acc);
            // 状态左移一格，把原始输入 xc 推进最新槽位（存 silu 前的值，与 HF 一致）。
            for (int i = 0; i < state_len - 1; ++i) st[i] = st[i + 1];
            st[state_len - 1] = xc;
        }
    }

    void gdn_step_ref(float *S, const float *q, const float *k, const float *v, float g,
                      float beta, float *o, int qk_dim, int v_dim) {
        // v_dim 上限给足（Qwen3.5-0.8B = 128），避免每 token 分配临时内存。
        constexpr int kMaxV = 512;
        if (v_dim <= 0 || v_dim > kMaxV) {
            std::fprintf(stderr, "gdn_step_ref: unsupported v_dim %d\n", v_dim);
            std::abort();
        }

        const float decay = std::exp(g);

        // 第 1 遍：S *= decay，同时求 kv_mem = S(旧)^T @ k。
        // 注意 kv_mem 必须用衰减后的状态（HF 先乘 g_t 再读）。
        float kv_mem[kMaxV];
        for (int d = 0; d < v_dim; ++d) kv_mem[d] = 0.0f;
        for (int j = 0; j < qk_dim; ++j) {
            float *Srow = S + static_cast<size_t>(j) * v_dim;
            const float kj = k[j];
            for (int d = 0; d < v_dim; ++d) {
                Srow[d] *= decay;
                kv_mem[d] += Srow[d] * kj;
            }
        }

        // 第 2 遍：delta = beta * (v - kv_mem)；S += outer(k, delta)。
        float delta[kMaxV];
        for (int d = 0; d < v_dim; ++d) delta[d] = beta * (v[d] - kv_mem[d]);
        for (int j = 0; j < qk_dim; ++j) {
            float *Srow = S + static_cast<size_t>(j) * v_dim;
            const float kj = k[j];
            for (int d = 0; d < v_dim; ++d) Srow[d] += kj * delta[d];
        }

        // 第 3 遍：o = S^T @ q（double 累加，与其他 ref kernel 一致）。
        for (int d = 0; d < v_dim; ++d) {
            double acc = 0.0;
            for (int j = 0; j < qk_dim; ++j) {
                acc += static_cast<double>(S[static_cast<size_t>(j) * v_dim + d]) *
                       static_cast<double>(q[j]);
            }
            o[d] = static_cast<float>(acc);
        }
    }

    void rmsnorm_gated_ref(const float *x, const float *gate, const float *weight, float *y,
                           int n, float eps) {
        double sumsq = 0.0;
        for (int i = 0; i < n; ++i) {
            const double xi = static_cast<double>(x[i]);
            sumsq += xi * xi;
        }
        const float scale = 1.0f / std::sqrt(static_cast<float>(sumsq / n) + eps);
        for (int i = 0; i < n; ++i) {
            const float normed = x[i] * scale * weight[i];
            y[i] = normed * silu_scalar(gate[i]);
        }
    }

    void partial_rope_ref(float *q, float *k, int n_heads, int n_kv_heads, int head_dim,
                          int rotary_dim, int pos, float theta) {
        if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
            std::fprintf(stderr, "partial_rope_ref: bad rotary_dim %d (head_dim %d)\n",
                         rotary_dim, head_dim);
            std::abort();
        }
        const int half = rotary_dim / 2;
        constexpr int kMaxHalf = 256;
        if (half > kMaxHalf) {
            std::fprintf(stderr, "partial_rope_ref: rotary_dim %d too large\n", rotary_dim);
            std::abort();
        }

        // 频率按 rotary_dim（而不是 head_dim）归一化——这是 partial RoPE 的关键。
        float cs[kMaxHalf];
        float sn[kMaxHalf];
        for (int i = 0; i < half; ++i) {
            const float exponent = -static_cast<float>(2 * i) / static_cast<float>(rotary_dim);
            const float inv_freq = std::pow(theta, exponent);
            const float angle = static_cast<float>(pos) * inv_freq;
            cs[i] = std::cos(angle);
            sn[i] = std::sin(angle);
        }

        // 只旋转每个 head 的前 rotary_dim 个分量（rotate-half 配对），
        // [rotary_dim, head_dim) 区间原样保留。
        const auto apply = [&](float *x) {
            for (int i = 0; i < half; ++i) {
                const float x0 = x[i];
                const float x1 = x[i + half];
                const float c = cs[i];
                const float s = sn[i];
                x[i] = x0 * c - x1 * s;
                x[i + half] = x1 * c + x0 * s;
            }
        };
        for (int h = 0; h < n_heads; ++h) {
            apply(q + static_cast<size_t>(h) * head_dim);
        }
        for (int h = 0; h < n_kv_heads; ++h) {
            apply(k + static_cast<size_t>(h) * head_dim);
        }
    }
} // namespace tinyqwen
