// QwenModel::forward_token 的实现：单 token 前向（decode 路径）
//
// 从 qwen_model.cpp 拆分出来，包含 Qwen2.x 和 Qwen3.5 的完整 forward 逻辑。

#include "qwen_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "dispatch.h"  // matvec_f32, argmax
#include "gdn_ops.h"
#include "ref_ops.h"

namespace tinyqwen {
    namespace {
        // HF GDN 的门控标量运算（fp32）：
        //   beta = sigmoid(b)；g = -exp(A_log) * softplus(a + dt_bias)。
        inline float sigmoidf32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

        inline float softplusf32(float x) {
            return x > 20.0f ? x : std::log1p(std::exp(x));
        }
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

    int QwenModel::forward_token(int token_id, TopKResult *topk, int topk_k) {
        const int hidden = static_cast<int>(cfg_.hidden_size);
        const int inter = static_cast<int>(cfg_.intermediate_size);
        const int vocab = static_cast<int>(cfg_.vocab_size);
        const int n_heads = static_cast<int>(cfg_.n_heads);
        const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
        const int head_dim = static_cast<int>(cfg_.head_dim);
        const int pos = kv_.seq_len();

        // 边界检查：token 必须在词表范围内、位置不能超容量。
        if (token_id < 0 || token_id >= vocab) {
            std::fprintf(stderr, "tinyqwen: token_id %d out of range [0, %d)\n", token_id, vocab);
            std::abort();
        }
        if (pos >= max_seq_len_) {
            std::fprintf(stderr, "tinyqwen: position %d exceeds max_seq_len %d\n", pos, max_seq_len_);
            std::abort();
        }

        Profiler &prof = *profiler_;
        prof.begin_token(token_count_, pos, token_count_ < prompt_len_);

        // profiler 作用域名（"layer_<i>.<op>"）写进同一个栈上 buffer，
        // 避免每个 op 都分配一次 std::string（那会很浪费）。
        char name[64];
        const auto scope = [&](const char *fmt, int layer) {
            std::snprintf(name, sizeof(name), fmt, layer);
            return name;
        };

        // ==== 第 1 步：词嵌入 ====
        // 用 token_id 当行号，从词嵌入表里取出对应那一行，作为残差流初值。
        // embed 布局 [vocab, hidden]，第 token_id 行起点 = embed_ + token_id*hidden。
        {
            ScopedTimer t(prof, "embed");
            if (dtype_ == Dtype::kF32 || dtype_ == Dtype::kI4) {
                // f32 文件或 I4 文件（embed 存为 fp32 lookup table）：直接 memcpy。
                std::memcpy(hidden_.data(),
                            static_cast<const float *>(embed_) +
                                    static_cast<size_t>(token_id) * hidden,
                            hidden * sizeof(float));
            } else {
                // f16：嵌入行转回 fp32 进残差流（hidden 流全程保持 fp32，
                // 只有权重是半精度）。每 token 只转一行（896 元素），开销可忽略。
                const uint16_t *row = static_cast<const uint16_t *>(embed_) +
                                      static_cast<size_t>(token_id) * hidden;
                for (int j = 0; j < hidden; ++j) hidden_[j] = half_to_float(row[j]);
            }
        }

        // attention 的缩放系数 1/sqrt(head_dim)。注意是 head_dim，不是 hidden——常见易错点。
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // ==== 第 2 步：逐层 transformer ====
        // token mixer 按模型/层类型三选一：
        //   - Qwen2.x          ：full attention（qkv 带 bias，整头 RoPE）；
        //   - Qwen3.5 full 层  ：full attention（无 bias、QK-norm、partial RoPE、
        //                        sigmoid 输出门，gate 从 q_proj 后半取出）；
        //   - Qwen3.5 linear 层：Gated DeltaNet（conv1d + delta rule 递归，O(1) 状态）。
        // FFN（SwiGLU）与残差结构两种架构完全一致，在分支之后共享。
        const bool is_qwen35 = cfg_.model_type == ModelType::kQwen35;
        for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
            const LayerWeights &w = layers_[i];

            // 2a. attention 前的 RMSNorm。
            // （Qwen3.5 是 zero-centered RMSNorm：导出端已把 +1 折进权重，
            // 所以这里和 Qwen2.x 共用同一个 kernel。）
            {
                ScopedTimer t(prof, scope("layer_%d.input_layernorm", i));
                backend_->rmsnorm(hidden_.data(), w.input_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }

            if (is_qwen35 && cfg_.is_linear_layer(i)) {
                // ---- GDN（linear attention）路径 ----
                const int n_v_heads = static_cast<int>(cfg_.linear_num_v_heads);
                const int n_qk_heads = static_cast<int>(cfg_.linear_num_qk_heads);
                const int qk_hd = static_cast<int>(cfg_.linear_qk_head_dim);
                const int v_hd = static_cast<int>(cfg_.linear_v_head_dim);
                const int key_dim = gdn_qk_dim_;
                const int li = cfg_.linear_layer_cache_index(i);
                float *mixed = mixed_.data();

                // 2b. 投影：混合 qkv + 门控 z + 标量 b/a（共用同一个 normed 输入）。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_proj", i));
                    mv(w.gdn_in_qkv, normed_.data(), mixed, gdn_conv_dim_, hidden);
                    mv(w.gdn_in_z, normed_.data(), z_.data(), gdn_value_dim_, hidden);
                    mv(w.gdn_in_b, normed_.data(), b_.data(), n_v_heads, hidden);
                    mv(w.gdn_in_a, normed_.data(), a_.data(), n_v_heads, hidden);
                }
                // 2c. causal conv1d 单步更新（就地；conv 后接 silu；状态推进）。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_conv1d", i));
                    backend_->causal_conv1d_update(mixed, gdn_state_.conv(li), w.gdn_conv_w, mixed,
                                         gdn_conv_dim_,
                                         static_cast<int>(cfg_.linear_conv_kernel_dim));
                }
                // 2d. q/k 逐头 l2norm + q 缩放（use_qk_l2norm_in_kernel=True）。
                //     先统一归一化，避免 rep>1 时同一 qk 头被重复归一化产生误差。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_l2norm", i));
                    const float q_scale = 1.0f / std::sqrt(static_cast<float>(qk_hd));
                    for (int h = 0; h < n_qk_heads; ++h) {
                        float *qh = mixed + h * qk_hd;
                        float *kh = mixed + key_dim + h * qk_hd;
                        backend_->l2norm_inplace(qh, qk_hd, 1e-6f);
                        for (int d = 0; d < qk_hd; ++d) qh[d] *= q_scale;
                        backend_->l2norm_inplace(kh, qk_hd, 1e-6f);
                    }
                }
                // 2e. gated delta rule 递归步：按 v 头更新状态并产出输出。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_recurrent", i));
                    float *S = gdn_state_.recurrent(li);
                    const int rep = n_v_heads / n_qk_heads;
                    for (int h = 0; h < n_v_heads; ++h) {
                        const int qk_h = h / rep;
                        const float *qh = mixed + qk_h * qk_hd;
                        const float *kh = mixed + key_dim + qk_h * qk_hd;
                        const float *vh = mixed + 2 * key_dim + h * v_hd;
                        const float g = -std::exp(w.gdn_a_log[h]) *
                                        softplusf32(a_[h] + w.gdn_dt_bias[h]);
                        const float beta = sigmoidf32(b_[h]);
                        backend_->gdn_step(S + static_cast<size_t>(h) * qk_hd * v_hd, qh, kh, vh, g,
                                 beta, gdn_out_.data() + h * v_hd, qk_hd, v_hd);
                    }
                }
                // 2f. 门控 RMSNorm：y = rmsnorm(o) * w * silu(z)，按 v 头维度。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_norm", i));
                    for (int h = 0; h < n_v_heads; ++h) {
                        backend_->rmsnorm_gated(gdn_out_.data() + h * v_hd, z_.data() + h * v_hd,
                                      w.gdn_norm, gdn_out_.data() + h * v_hd, v_hd,
                                      cfg_.rms_norm_eps);
                    }
                }
                // 2g. 输出投影 + 残差。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_out_proj", i));
                    mv(w.gdn_out_proj, gdn_out_.data(), o_.data(), hidden, gdn_value_dim_);
                }
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
                }
            } else if (is_qwen35) {
                // ---- Qwen3.5 full attention 路径 ----
                const int ci = cfg_.full_layer_cache_index(i);

                // 2b. q(+gate)/k/v 投影。q_proj 形状 [2*q_dim, hidden]，
                //     按 head 交错：每头前 head_dim 是 query，后 head_dim 是输出门。
                {
                    ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                    mv(w.q_proj, normed_.data(), q_full_.data(), 2 * q_dim_, hidden);
                    for (int h = 0; h < n_heads; ++h) {
                        const float *src = q_full_.data() + h * 2 * head_dim;
                        std::memcpy(q_.data() + h * head_dim, src, head_dim * sizeof(float));
                        std::memcpy(q_gate_.data() + h * head_dim, src + head_dim,
                                    head_dim * sizeof(float));
                    }
                    if (fuse_qkv_) {
                        mv_pair(w.k_proj, w.v_proj, normed_.data(), k_.data(), v_.data(),
                                kv_dim_, hidden);
                    } else {
                        mv(w.k_proj, normed_.data(), k_.data(), kv_dim_, hidden);
                        mv(w.v_proj, normed_.data(), v_.data(), kv_dim_, hidden);
                    }
                }
                // 2c. QK per-head RMSNorm（zero-centered，权重已折 +1）。
                {
                    ScopedTimer t(prof, scope("layer_%d.qk_norm", i));
                    for (int h = 0; h < n_heads; ++h) {
                        backend_->rmsnorm(q_.data() + h * head_dim, w.q_norm, q_.data() + h * head_dim,
                                head_dim, cfg_.rms_norm_eps);
                    }
                    for (int h = 0; h < n_kv_heads; ++h) {
                        backend_->rmsnorm(k_.data() + h * head_dim, w.k_norm, k_.data() + h * head_dim,
                                head_dim, cfg_.rms_norm_eps);
                    }
                }
                // 2d. partial RoPE：只旋转每头前 rotary_dim 维（rotate-half）。
                {
                    ScopedTimer t(prof, scope("layer_%d.rope", i));
                    backend_->partial_rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim,
                                     rotary_dim_, pos, cfg_.rope_theta);
                }
                // 2e. k/v 追加进 cache（紧凑下标 ci：只存 full 层）。
                {
                    ScopedTimer t(prof, scope("layer_%d.kv_append", i));
                    const size_t pos_off = static_cast<size_t>(pos) * head_dim;
                    const size_t head_plane = static_cast<size_t>(max_seq_len_) * head_dim;
                    float *k_layer = kv_.k(ci);
                    float *v_layer = kv_.v(ci);
                    for (int h = 0; h < n_kv_heads; ++h) {
                        std::memcpy(k_layer + h * head_plane + pos_off,
                                    k_.data() + h * head_dim, head_dim * sizeof(float));
                        std::memcpy(v_layer + h * head_plane + pos_off,
                                    v_.data() + h * head_dim, head_dim * sizeof(float));
                    }
                }
                // 2f. attention（GQA）+ sigmoid 输出门。
                {
                    ScopedTimer t(prof, scope("layer_%d.attention", i));
                    backend_->attention_decode(q_.data(), kv_.k(ci), kv_.v(ci), pos + 1, max_seq_len_,
                                     n_heads, n_kv_heads, head_dim, attn_scale, attn_.data());
                    for (int j = 0; j < q_dim_; ++j) attn_[j] *= sigmoidf32(q_gate_[j]);
                }
                // 2g. o_proj + 残差。
                {
                    ScopedTimer t(prof, scope("layer_%d.o_proj", i));
                    mv(w.o_proj, attn_.data(), o_.data(), hidden, q_dim_);
                }
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
                }
            } else {
                // ---- Qwen2.x attention 路径（与 v1 完全一致）----
                // 2b. q/k/v 投影 + bias。Qwen2/2.5 的 q/k/v 有 bias，且必须加在 RoPE 之前。
                if (fuse_qkv_) {
                    ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                    mv_qkv(w.q_proj, w.k_proj, w.v_proj, normed_.data(),
                            q_.data(), k_.data(), v_.data(), q_dim_, kv_dim_, hidden);
                    for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
                    for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                    for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
                } else {
                    {
                        ScopedTimer t(prof, scope("layer_%d.q_proj", i));
                        mv(w.q_proj, normed_.data(), q_.data(), q_dim_, hidden);
                        for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
                    }
                    {
                        ScopedTimer t(prof, scope("layer_%d.kv_proj", i));
                        mv_pair(w.k_proj, w.v_proj, normed_.data(), k_.data(), v_.data(),
                                kv_dim_, hidden);
                        for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                        for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
                    }
                }
                // 2c. RoPE 旋转位置编码：把"位置 pos"的信息编进 q/k（v 不需要）。
                {
                    ScopedTimer t(prof, scope("layer_%d.rope", i));
                    backend_->rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim, pos,
                         cfg_.rope_theta);
                }
                // 2d. 把当前 token 的 k/v 追加进 cache。注意必须先 append 再 attend：
                // 当前 token 要能"看到"自己，所以下面 attention 读的 seq_len = pos+1。
                {
                    ScopedTimer t(prof, scope("layer_%d.kv_append", i));
                    const size_t pos_off = static_cast<size_t>(pos) * head_dim;
                    const size_t head_plane = static_cast<size_t>(max_seq_len_) * head_dim;
                    float *k_layer = kv_.k(static_cast<int>(i));
                    float *v_layer = kv_.v(static_cast<int>(i));
                    for (int h = 0; h < n_kv_heads; ++h) {
                        std::memcpy(k_layer + h * head_plane + pos_off, k_.data() + h * head_dim,
                                    head_dim * sizeof(float));
                        std::memcpy(v_layer + h * head_plane + pos_off, v_.data() + h * head_dim,
                                    head_dim * sizeof(float));
                    }
                }
                // 2e. attention：当前 q 对 cache 里 [0..pos] 所有位置加权求和。
                {
                    ScopedTimer t(prof, scope("layer_%d.attention", i));
                    backend_->attention_decode(q_.data(), kv_.k(static_cast<int>(i)),
                                     kv_.v(static_cast<int>(i)), pos + 1, max_seq_len_, n_heads,
                                     n_kv_heads, head_dim, attn_scale, attn_.data());
                }
                // 2f. 输出投影 o_proj。
                {
                    ScopedTimer t(prof, scope("layer_%d.o_proj", i));
                    mv(w.o_proj, attn_.data(), o_.data(), hidden, q_dim_);
                }
                // 2g. 第一次残差连接：x = x + attention(x)。残差让梯度/信息能直通。
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
                }
            }

            // ---- FFN 块（SwiGLU）：norm -> gate/up -> silu*up -> down ----

            // 2h. FFN 前的 RMSNorm。
            {
                ScopedTimer t(prof, scope("layer_%d.post_attn_layernorm", i));
                backend_->rmsnorm(hidden_.data(), w.post_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }
            // 2i. gate 和 up 两个投影（SwiGLU 需要两条支路）。
            if (fuse_gate_up_) {
                ScopedTimer t(prof, scope("layer_%d.gate_up_proj", i));
                mv_pair(w.gate, w.up, normed_.data(), gate_.data(), up_.data(), inter, hidden);
            } else {
                {
                    ScopedTimer t(prof, scope("layer_%d.gate_proj", i));
                    mv(w.gate, normed_.data(), gate_.data(), inter, hidden);
                }
                {
                    ScopedTimer t(prof, scope("layer_%d.up_proj", i));
                    mv(w.up, normed_.data(), up_.data(), inter, hidden);
                }
            }
            // 2j. SwiGLU 融合：SiLU 只作用在 gate 支路，再和 up 逐元素相乘。
            // gate_ 就地复用为融合结果，直接喂给 down_proj。swiglu() 走 ops
            // dispatch：默认兜底 swiglu_ref（= 原两步逐位一致），注册了 NEON
            // 变体时单遍向量化。
            {
                ScopedTimer t(prof, scope("layer_%d.swiglu", i));
                backend_->swiglu(gate_.data(), up_.data(), inter);
            }
            // 2k. down 投影，把维度从 inter 压回 hidden。
            {
                ScopedTimer t(prof, scope("layer_%d.down_proj", i));
                mv(w.down, gate_.data(), ffn_.data(), hidden, inter);
            }
            // 2l. 第二次残差连接：x = x + ffn(x)。
            {
                ScopedTimer t(prof, scope("layer_%d.residual_ffn", i));
                for (int j = 0; j < hidden; ++j) hidden_[j] += ffn_[j];
            }
        }

        // ==== 第 3 步：最后的 norm + 投影到词表 ====
        {
            ScopedTimer t(prof, "final_norm");
            backend_->rmsnorm(hidden_.data(), final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
        }
        {
            // lm_head：把 hidden 向量投成 vocab 维的 logits（每个词一个分数）。
            // tied 时 lm_head_ 就是 embed_（见 create）。I4 tied 时 embed 是 fp32。
            ScopedTimer t(prof, "lm_head");
            if (lm_head_is_f32_) {
                matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                           logits_.data(), vocab, hidden);
            } else {
                mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
            }
        }

        // ==== 第 4 步：greedy 取 argmax ====
        int next = 0;
        {
            // 要 top-k 就顺便取，argmax 就是 top-1，不重复扫。
            ScopedTimer t(prof, "topk_argmax");
            if (topk) {
                top_k_logits(logits_.data(), vocab, topk_k, topk);
                next = topk->indices.empty() ? 0 : topk->indices[0];
            } else {
                next = argmax(logits_.data(), vocab);
            }
        }

        // 提交本 token 写入的 cache 槽位，收尾 profiler 记录。
        kv_.advance(1);
        token_count_ += 1;
        prof.end_token();
        return next;
    }
} // namespace tinyqwen
