// QwenModel::forward_prefill 的实现：批量 prefill（GEMM + per-token attention）
//
// 从 qwen_model.cpp 拆分出来，只支持 Qwen2.x（Qwen3.5 回退到 forward_token）。

#include "qwen_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "dispatch.h"  // matvec_f32, argmax
#include "ref_ops.h"

namespace tinyqwen {
    namespace {
        // 取分数最高的 k 个 token。实现：对下标数组做 partial_sort，
        // 只把前 k 个排好序，复杂度 O(vocab * k)。结果第一个就是 argmax。
        void top_k_logits(const float *logits, int vocab, int k, TopKResult *out) {
            k = std::min(k, vocab);
            std::vector<int> idx(vocab);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [logits](int a, int b) { return logits[a] > logits[b]; });
            out->indices.resize(k);
            out->values.resize(k);
            for (int i = 0; i < k; ++i) {
                out->indices[i] = idx[i];
                out->values[i] = logits[idx[i]];
            }
        }
    } // namespace

    int QwenModel::forward_prefill(const int *token_ids, int n,
                                   TopKResult *topk, int topk_k) {
        if (n <= 0) return -1;
        if (n == 1) return forward_token(token_ids[0], topk, topk_k);

        const int hidden = static_cast<int>(cfg_.hidden_size);
        const int inter = static_cast<int>(cfg_.intermediate_size);
        const int vocab = static_cast<int>(cfg_.vocab_size);
        const int n_heads = static_cast<int>(cfg_.n_heads);
        const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
        const int head_dim = static_cast<int>(cfg_.head_dim);
        const int base_pos = kv_.seq_len();
        const bool is_qwen35 = cfg_.model_type == ModelType::kQwen35;

        // GDN (linear attention) layers need sequential state updates — fall back
        if (is_qwen35) {
            int last = -1;
            for (int i = 0; i < n; ++i)
                last = forward_token(token_ids[i], (i == n - 1) ? topk : nullptr, topk_k);
            return last;
        }

        Profiler &prof = *profiler_;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Allocate batch buffers (column-major: each col = one token's vector)
        const size_t N = static_cast<size_t>(n);
        std::vector<float> hid_batch(hidden * N);   // residual stream [hidden, N]
        std::vector<float> norm_batch(hidden * N);  // after rmsnorm
        std::vector<float> q_batch(q_dim_ * N);
        std::vector<float> k_batch(kv_dim_ * N);
        std::vector<float> v_batch(kv_dim_ * N);
        std::vector<float> attn_batch(q_dim_ * N);
        std::vector<float> o_batch(hidden * N);
        std::vector<float> gate_batch(inter * N);
        std::vector<float> up_batch(inter * N);
        std::vector<float> ffn_batch(hidden * N);

        // Step 1: Embed all tokens
        {
            ScopedTimer t(prof, "prefill_embed");
            for (int c = 0; c < n; ++c) {
                int tid = token_ids[c];
                float *dst = hid_batch.data() + static_cast<size_t>(c) * hidden;
                if (dtype_ == Dtype::kF32 || dtype_ == Dtype::kI4) {
                    std::memcpy(dst, static_cast<const float *>(embed_) +
                                    static_cast<size_t>(tid) * hidden,
                                hidden * sizeof(float));
                } else {
                    const uint16_t *row = static_cast<const uint16_t *>(embed_) +
                                          static_cast<size_t>(tid) * hidden;
                    for (int j = 0; j < hidden; ++j) dst[j] = half_to_float(row[j]);
                }
            }
        }

        // Step 2: Layer loop
        char name[64];
        for (uint32_t li = 0; li < cfg_.n_layers; ++li) {
            const LayerWeights &w = layers_[li];

            // 2a. RMSNorm all tokens
            {
                std::snprintf(name, sizeof(name), "layer_%d.input_layernorm", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.input_ln,
                            norm_batch.data() + static_cast<size_t>(c) * hidden,
                            hidden, cfg_.rms_norm_eps);
                }
            }

            // 2b. Q/K/V projections via GEMM
            {
                std::snprintf(name, sizeof(name), "layer_%d.qkv_proj", li);
                ScopedTimer t(prof, name);
                mm(w.q_proj, norm_batch.data(), q_batch.data(), q_dim_, hidden, n);
                mm(w.k_proj, norm_batch.data(), k_batch.data(), kv_dim_, hidden, n);
                mm(w.v_proj, norm_batch.data(), v_batch.data(), kv_dim_, hidden, n);
            }

            // 2c. Per-token: bias + RoPE + KV append + attention
            {
                std::snprintf(name, sizeof(name), "layer_%d.attn_loop", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    const int pos = base_pos + c;
                    float *qc = q_batch.data() + static_cast<size_t>(c) * q_dim_;
                    float *kc = k_batch.data() + static_cast<size_t>(c) * kv_dim_;
                    float *vc = v_batch.data() + static_cast<size_t>(c) * kv_dim_;
                    float *ac = attn_batch.data() + static_cast<size_t>(c) * q_dim_;

                    // bias
                    if (w.q_bias) for (int j = 0; j < q_dim_; ++j) qc[j] += w.q_bias[j];
                    if (w.k_bias) for (int j = 0; j < kv_dim_; ++j) kc[j] += w.k_bias[j];
                    if (w.v_bias) for (int j = 0; j < kv_dim_; ++j) vc[j] += w.v_bias[j];

                    // RoPE
                    backend_->rope(qc, kc, n_heads, n_kv_heads, head_dim, pos, cfg_.rope_theta);

                    // KV append
                    const size_t pos_off = static_cast<size_t>(pos) * head_dim;
                    const size_t head_plane = static_cast<size_t>(max_seq_len_) * head_dim;
                    float *k_layer = kv_.k(static_cast<int>(li));
                    float *v_layer = kv_.v(static_cast<int>(li));
                    for (int h = 0; h < n_kv_heads; ++h) {
                        std::memcpy(k_layer + h * head_plane + pos_off,
                                    kc + h * head_dim, head_dim * sizeof(float));
                        std::memcpy(v_layer + h * head_plane + pos_off,
                                    vc + h * head_dim, head_dim * sizeof(float));
                    }

                    // Attention
                    backend_->attention_decode(qc, kv_.k(static_cast<int>(li)),
                                     kv_.v(static_cast<int>(li)), pos + 1, max_seq_len_,
                                     n_heads, n_kv_heads, head_dim, attn_scale, ac);
                }
            }

            // 2d. O projection via GEMM + residual
            {
                std::snprintf(name, sizeof(name), "layer_%d.o_proj", li);
                ScopedTimer t(prof, name);
                mm(w.o_proj, attn_batch.data(), o_batch.data(), hidden, q_dim_, n);
            }
            {
                std::snprintf(name, sizeof(name), "layer_%d.residual_attn", li);
                ScopedTimer t(prof, name);
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    hid_batch[j] += o_batch[j];
            }

            // 2e. Post-attention RMSNorm
            {
                std::snprintf(name, sizeof(name), "layer_%d.post_attn_layernorm", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.post_ln,
                            norm_batch.data() + static_cast<size_t>(c) * hidden,
                            hidden, cfg_.rms_norm_eps);
                }
            }

            // 2f. FFN gate/up via GEMM
            {
                std::snprintf(name, sizeof(name), "layer_%d.gate_up_proj", li);
                ScopedTimer t(prof, name);
                mm(w.gate, norm_batch.data(), gate_batch.data(), inter, hidden, n);
                mm(w.up, norm_batch.data(), up_batch.data(), inter, hidden, n);
            }

            // 2g. SwiGLU (per-token)
            {
                std::snprintf(name, sizeof(name), "layer_%d.swiglu", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->swiglu(gate_batch.data() + static_cast<size_t>(c) * inter,
                           up_batch.data() + static_cast<size_t>(c) * inter, inter);
                }
            }

            // 2h. Down projection via GEMM + residual
            {
                std::snprintf(name, sizeof(name), "layer_%d.down_proj", li);
                ScopedTimer t(prof, name);
                mm(w.down, gate_batch.data(), ffn_batch.data(), hidden, inter, n);
            }
            {
                std::snprintf(name, sizeof(name), "layer_%d.residual_ffn", li);
                ScopedTimer t(prof, name);
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    hid_batch[j] += ffn_batch[j];
            }
        }

        // Step 3: Final norm + lm_head (last token only)
        const float *last_hidden = hid_batch.data() + static_cast<size_t>(n - 1) * hidden;
        {
            ScopedTimer t(prof, "final_norm");
            backend_->rmsnorm(last_hidden, final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
        }
        {
            ScopedTimer t(prof, "lm_head");
            if (lm_head_is_f32_) {
                matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                           logits_.data(), vocab, hidden);
            } else {
                mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
            }
        }

        // Step 4: argmax
        int next = 0;
        {
            ScopedTimer t(prof, "topk_argmax");
            if (topk) {
                top_k_logits(logits_.data(), vocab, topk_k, topk);
                next = topk->indices.empty() ? 0 : topk->indices[0];
            } else {
                next = argmax(logits_.data(), vocab);
            }
        }

        // Advance KV cache and token counter for all N tokens
        kv_.advance(n);
        token_count_ += n;
        return next;
    }
} // namespace tinyqwen
