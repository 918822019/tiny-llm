// ============================================================================
// qwen_forward_token.cpp — 单 token 前向实现（decode 路径）
// ============================================================================
// 本文件实现 QwenModel::forward_token() 方法，负责对单个 token 做一次完整
// 前向计算。这是 decode 阶段的路径，也是 prefill 逐 token 模式（verbose 和
// GPU engine）的路径。
//
// 支持三种模型架构：
//   1. Qwen2.x           — full attention + qkv bias + 全头 RoPE
//   2. Qwen3.5 full 层   — full attention + QK-norm + partial RoPE + sigmoid
//                          输出门（gate 从 q_proj 后半取出）
//   3. Qwen3.5 linear 层 — Gated DeltaNet（causal conv1d + delta rule 递归，
//                          O(1) 状态复杂度）
//
// 一个 token 的完整前向流程：
//   Step 1: 词嵌入 - 查表获取嵌入向量
//   Step 2: 逐层 Transformer
//     - Token mixer（按模型/层类型三选一）：
//       a) Qwen2.x full attention：qkv_proj(+bias) -> rope -> kv_append -> attention -> o_proj
//       b) Qwen3.5 full attention：qkv_proj -> qk_norm -> partial_rope -> kv_append -> attention -> sigmoid gate -> o_proj
//       c) Qwen3.5 GDN：in_proj -> causal_conv1d -> l2norm -> delta_rule -> rmsnorm_gated -> out_proj
//     - FFN 块（SwiGLU）：post_norm -> gate/up_proj -> swiglu -> down_proj + residual
//   Step 3: 最终 norm + lm_head 投影
//   Step 4: argmax 取下一个 token
// ============================================================================

#include "qwen_model.h"

#include <algorithm>   // std::min, std::partial_sort
#include <cmath>       // std::sqrt, std::exp, std::log1p
#include <cstdio>      // 标准输入输出
#include <cstring>     // 内存操作
#include <numeric>     // std::iota

#include "dispatch.h"  // matvec_f32, argmax
#include "gdn_ops.h"   // Qwen3.5 GDN 算子
#include "ref_ops.h"   // ref 实现

namespace tinyqwen {
    namespace {
        // =====================================================================
        // GDN 门控标量运算（fp32）
        // =====================================================================
        // 这些函数用于 GDN 层的门控参数计算，对应 HuggingFace 实现中的
        // beta = sigmoid(b) 和 g = -exp(A_log) * softplus(a + dt_bias)。

        // sigmoid(x) = 1 / (1 + exp(-x))：标准 logistic 函数
        inline float sigmoidf32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

        // softplus(x) = log(1 + exp(x))：ReLU 的平滑近似
        // 当 x > 20 时，exp(x) 可能溢出，所以用 x 近似（log(1 + exp(x)) ≈ x）
        inline float softplusf32(float x) {
            return x > 20.0f ? x : std::log1p(std::exp(x));
        }

        // 取分数最高的 k 个 token（使用 partial_sort，复杂度 O(vocab * k)）
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

    // =========================================================================
    // QwenModel::forward_token() — 单 token 前向
    // =========================================================================
    // 参数：
    //   token_id — 当前 token 的 ID
    //   topk     — 输出参数，不为 nullptr 时写入 top-k 结果
    //   topk_k   — top-k 的 k 值
    // 返回值：下一个要生成的 token ID
    // 说明：pos 是当前 KV 长度（= 这是序列里第几个位置）；调用结束后 cache
    //       有 pos+1 条，返回值是 greedy 的下一个 token。
    //       数学定义见 docs/qwen_forward.md。
    int QwenModel::forward_token(int token_id, TopKResult *topk, int topk_k,
                                 bool need_logits) {
        // 提取配置参数（转为 int 便于循环中使用）
        const int hidden = static_cast<int>(cfg_.hidden_size);
        const int inter = static_cast<int>(cfg_.intermediate_size);
        const int vocab = static_cast<int>(cfg_.vocab_size);
        const int n_heads = static_cast<int>(cfg_.n_heads);
        const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
        const int head_dim = static_cast<int>(cfg_.head_dim);
        const int pos = kv_.seq_len(); // 当前 KV cache 中已存储的序列长度

        // 边界检查：token 必须在词表范围内、位置不能超容量
        if (token_id < 0 || token_id >= vocab) {
            std::fprintf(stderr, "tinyqwen: token_id %d out of range [0, %d)\n", token_id, vocab);
            std::abort();
        }
        if (pos >= max_seq_len_) {
            std::fprintf(stderr, "tinyqwen: position %d exceeds max_seq_len %d\n", pos, max_seq_len_);
            std::abort();
        }

        Profiler &prof = *profiler_;
        // 开始 profiling 记录：token_count_ < prompt_len_ 表示是 prefill 阶段
        prof.begin_token(token_count_, pos, token_count_ < prompt_len_);

        // profiler 作用域名（"layer_<i>.<op>"）写进同一个栈上 buffer，
        // 避免每个 op 都分配一次 std::string（那会很浪费）
        char name[64];
        const auto scope = [&](const char *fmt, int layer) {
            std::snprintf(name, sizeof(name), fmt, layer);
            return name;
        };

        // =====================================================================
        // Step 1: 词嵌入
        // =====================================================================
        // 用 token_id 当行号，从词嵌入表里取出对应那一行，作为残差流初值。
        // embed 布局 [vocab, hidden]，第 token_id 行起点 = embed_ + token_id*hidden。
        {
            ScopedTimer t(prof, "embed");
            if (embed_dtype_ == Dtype::kF32) {
                // f32 文件或 I4 文件（embed 存为 fp32 lookup table）：直接 memcpy
                std::memcpy(hidden_.data(),
                            static_cast<const float *>(embed_) +
                                    static_cast<size_t>(token_id) * hidden,
                            hidden * sizeof(float));
            } else if (embed_dtype_ == Dtype::kI4) {
                // 紧凑 INT4 embed：反量化该行（每 token 只解一行，开销可忽略）
                dequant_i4_row(static_cast<const uint8_t *>(embed_), token_id, hidden,
                               group_size_, hidden_.data());
            } else {
                // f16：嵌入行转回 fp32 进残差流（hidden 流全程保持 fp32，
                // 只有权重是半精度）。每 token 只转一行（896 元素），开销可忽略
                const uint16_t *row = static_cast<const uint16_t *>(embed_) +
                                      static_cast<size_t>(token_id) * hidden;
                for (int j = 0; j < hidden; ++j) hidden_[j] = half_to_float(row[j]);
            }
        }

        // attention 的缩放系数 1/sqrt(head_dim)。注意是 head_dim，不是 hidden——常见易错点
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // =====================================================================
        // Step 2: 逐层 Transformer
        // =====================================================================
        // token mixer 按模型/层类型三选一：
        //   - Qwen2.x          ：full attention（qkv 带 bias，整头 RoPE）
        //   - Qwen3.5 full 层  ：full attention（无 bias、QK-norm、partial RoPE、
        //                        sigmoid 输出门，gate 从 q_proj 后半取出）
        //   - Qwen3.5 linear 层：Gated DeltaNet（conv1d + delta rule 递归，O(1) 状态）
        const bool is_qwen35 = cfg_.uses_qwen35_attention();
        for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
            const LayerWeights &w = layers_[i];

            // 2a. attention 前的 RMSNorm
            // （Qwen3.5 是 zero-centered RMSNorm：导出端已把 +1 折进权重，
            // 所以这里和 Qwen2.x 共用同一个 kernel。）
            {
                ScopedTimer t(prof, scope("layer_%d.input_layernorm", i));
                backend_->rmsnorm(hidden_.data(), w.input_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }

            if (is_qwen35 && cfg_.is_linear_layer(i)) {
                // ---- GDN（linear attention / Gated DeltaNet）路径 ----
                // 这是 Qwen3.5 混合架构的核心创新：用 O(1) 状态的线性注意力
                // 替代 O(n) 的 softmax attention，在长序列场景下显著降低内存和计算

                const int n_v_heads = static_cast<int>(cfg_.linear_num_v_heads);
                const int n_qk_heads = static_cast<int>(cfg_.linear_num_qk_heads);
                const int qk_hd = static_cast<int>(cfg_.linear_qk_head_dim);
                const int v_hd = static_cast<int>(cfg_.linear_v_head_dim);
                const int key_dim = gdn_qk_dim_; // qk 的总维度 = n_qk_heads * qk_head_dim
                const int li = cfg_.linear_layer_cache_index(i); // linear 层的本地索引
                float *mixed = mixed_.data();

                // 2b. 投影：混合 qkv + 门控 z + 标量 b/a（共用同一个 normed 输入）
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_proj", i));
                    // in_proj_qkv：同时投影 q、k、v（拼接为 conv_dim 维）
                    mv(w.gdn_in_qkv, normed_.data(), mixed, gdn_conv_dim_, hidden);
                    // in_proj_z：门控信号 z（用于 silu 门控）
                    mv(w.gdn_in_z, normed_.data(), z_.data(), gdn_value_dim_, hidden);
                    // in_proj_b：标量 b（用于 sigmoid 门控 beta）
                    mv(w.gdn_in_b, normed_.data(), b_.data(), n_v_heads, hidden);
                    // in_proj_a：标量 a（用于衰减因子 g）
                    mv(w.gdn_in_a, normed_.data(), a_.data(), n_v_heads, hidden);
                }

                // 2c. causal conv1d 单步更新（就地；conv 后接 silu；状态推进）
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_conv1d", i));
                    backend_->causal_conv1d_update(mixed, gdn_state_.conv(li), w.gdn_conv_w, mixed,
                                         gdn_conv_dim_,
                                         static_cast<int>(cfg_.linear_conv_kernel_dim));
                }

                // 2d. q/k 逐头 l2norm + q 缩放
                //     先统一归一化，避免 rep>1 时同一 qk 头被重复归一化产生误差
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_l2norm", i));
                    const float q_scale = 1.0f / std::sqrt(static_cast<float>(qk_hd));
                    for (int h = 0; h < n_qk_heads; ++h) {
                        // mixed 布局：[q (qk_dim)] [k (qk_dim)] [v (v_dim)]
                        float *qh = mixed + h * qk_hd;                    // query 头 h
                        float *kh = mixed + key_dim + h * qk_hd;          // key 头 h
                        // L2 归一化 query
                        backend_->l2norm_inplace(qh, qk_hd, 1e-6f);
                        // query 缩放（除以 sqrt(qk_hd)）
                        for (int d = 0; d < qk_hd; ++d) qh[d] *= q_scale;
                        // L2 归一化 key
                        backend_->l2norm_inplace(kh, qk_hd, 1e-6f);
                    }
                }

                // 2e. gated delta rule 递归步：按 v 头更新状态并产出输出
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_recurrent", i));
                    float *S = gdn_state_.recurrent(li); // 递归状态矩阵
                    const int rep = n_v_heads / n_qk_heads; // 每个 qk 头对应几个 v 头
                    for (int h = 0; h < n_v_heads; ++h) {
                        const int qk_h = h / rep; // 当前 v 头对应的 qk 头索引
                        const float *qh = mixed + qk_h * qk_hd;
                        const float *kh = mixed + key_dim + qk_h * qk_hd;
                        const float *vh = mixed + 2 * key_dim + h * v_hd;
                        // 衰减因子 g = -exp(A_log[h]) * softplus(a[h] + dt_bias[h])
                        const float g = -std::exp(w.gdn_a_log[h]) *
                                        softplusf32(a_[h] + w.gdn_dt_bias[h]);
                        // 门控因子 beta = sigmoid(b[h])
                        const float beta = sigmoidf32(b_[h]);
                        // 执行 delta rule 递归：S = g*S + outer(k,v); o = beta * S^T * q
                        backend_->gdn_step(S + static_cast<size_t>(h) * qk_hd * v_hd, qh, kh, vh, g,
                                 beta, gdn_out_.data() + h * v_hd, qk_hd, v_hd);
                    }
                }

                // 2f. 门控 RMSNorm：y = rmsnorm(o) * w * silu(z)，按 v 头维度
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_norm", i));
                    for (int h = 0; h < n_v_heads; ++h) {
                        backend_->rmsnorm_gated(gdn_out_.data() + h * v_hd, z_.data() + h * v_hd,
                                      w.gdn_norm, gdn_out_.data() + h * v_hd, v_hd,
                                      cfg_.rms_norm_eps);
                    }
                }

                // 2g. 输出投影 + 残差
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
                // 与 Qwen2.x 的关键区别：
                //   - 无 q/k/v bias
                //   - q_proj 形状 [2*q_dim, hidden]：前半是 query，后半是输出门
                //   - 使用 QK-norm（per-head RMSNorm）
                //   - partial RoPE（只旋转 rotary_dim 维）
                //   - sigmoid 输出门（替代标准的残差连接）
                const int ci = cfg_.full_layer_cache_index(i); // full 层的 KV cache 索引

                // 2b. q(+gate)/k/v 投影。q_proj 形状 [2*q_dim, hidden]，
                //     按 head 交错：每头前 head_dim 是 query，后 head_dim 是输出门
                {
                    ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                    // 一次投影得到 [query | gate]
                    mv(w.q_proj, normed_.data(), q_full_.data(), 2 * q_dim_, hidden);
                    // 按 head 交错分解：前半是 q，后半是 gate
                    for (int h = 0; h < n_heads; ++h) {
                        const float *src = q_full_.data() + h * 2 * head_dim;
                        std::memcpy(q_.data() + h * head_dim, src, head_dim * sizeof(float));
                        std::memcpy(q_gate_.data() + h * head_dim, src + head_dim,
                                    head_dim * sizeof(float));
                    }
                    // K 和 V 投影（可选融合）
                    if (fuse_qkv_) {
                        // 融合版本：K 和 V 一次投影完成
                        mv_pair(w.k_proj, w.v_proj, normed_.data(), k_.data(), v_.data(),
                                kv_dim_, hidden);
                    } else {
                        mv(w.k_proj, normed_.data(), k_.data(), kv_dim_, hidden);
                        mv(w.v_proj, normed_.data(), v_.data(), kv_dim_, hidden);
                    }
                }

                // 2c. QK per-head RMSNorm（zero-centered，权重已折 +1）
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

                // 2d. partial RoPE：只旋转每头前 rotary_dim 维（rotate-half）
                {
                    ScopedTimer t(prof, scope("layer_%d.rope", i));
                    backend_->partial_rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim,
                                     rotary_dim_, pos, cfg_.rope_theta);
                }

                // 2e. k/v 追加进 cache（紧凑下标 ci：只存 full 层）
                //     统一写入入口：内部按存储精度转换（fp32 memcpy / fp16 转换）
                {
                    ScopedTimer t(prof, scope("layer_%d.kv_append", i));
                    kv_.write_token(ci, pos, k_.data(), v_.data());
                }

                // 2f. attention（GQA）+ sigmoid 输出门
                {
                    ScopedTimer t(prof, scope("layer_%d.attention", i));
                    // attention 统一分发（按 KV 精度）+ sigmoid 输出门
                    attention_kv(q_.data(), ci, pos + 1, n_heads, n_kv_heads, head_dim,
                                 attn_scale, attn_.data());
                    // 应用 sigmoid 输出门：attn[i] *= sigmoid(gate[i])
                    for (int j = 0; j < q_dim_; ++j) attn_[j] *= sigmoidf32(q_gate_[j]);
                }

                // 2g. o_proj + 残差
                {
                    ScopedTimer t(prof, scope("layer_%d.o_proj", i));
                    mv(w.o_proj, attn_.data(), o_.data(), hidden, q_dim_);
                }
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
                }

            } else {
                // ---- Qwen2.x / Qwen3 稠密 attention 路径 ----
                // 2b. q/k/v 投影 + 可选 bias。Qwen2.x 有 bias（须加在 RoPE 之前），
                //     Qwen3 稠密 attention_bias=false 无 bias（权重为 nullptr）
                if (rotated_) {
                    // 旋转模型：q/k/v 旋转各不相同，不能融合，逐个旋转+投影
                    ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                    mv_rot(w.q_proj, normed_.data(), q_.data(), q_dim_, hidden, w.rot_q);
                    mv_rot(w.k_proj, normed_.data(), k_.data(), kv_dim_, hidden, w.rot_k);
                    mv_rot(w.v_proj, normed_.data(), v_.data(), kv_dim_, hidden, w.rot_v);
                    // bias 在输出侧，不受输入旋转影响
                    if (w.q_bias) for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
                    if (w.k_bias) for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                    if (w.v_bias) for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
                } else if (fuse_qkv_) {
                    // 融合 QKV 投影：一次计算三个投影
                    ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                    mv_qkv(w.q_proj, w.k_proj, w.v_proj, normed_.data(),
                            q_.data(), k_.data(), v_.data(), q_dim_, kv_dim_, hidden);
                    if (w.q_bias) for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
                    if (w.k_bias) for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                    if (w.v_bias) for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
                } else {
                    // 非融合版本：分别计算 Q、K、V 投影
                    {
                        ScopedTimer t(prof, scope("layer_%d.q_proj", i));
                        mv(w.q_proj, normed_.data(), q_.data(), q_dim_, hidden);
                        if (w.q_bias) for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
                    }
                    {
                        ScopedTimer t(prof, scope("layer_%d.kv_proj", i));
                        mv_pair(w.k_proj, w.v_proj, normed_.data(), k_.data(), v_.data(),
                                kv_dim_, hidden);
                        if (w.k_bias) for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                        if (w.v_bias) for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
                    }
                }

                // 2b'. QK per-head RMSNorm（Qwen3 稠密有，Qwen2.x 无；须在 RoPE 之前）
                if (w.q_norm) {
                    ScopedTimer t(prof, scope("layer_%d.qk_norm", i));
                    for (int h = 0; h < n_heads; ++h)
                        backend_->rmsnorm(q_.data() + h * head_dim, w.q_norm,
                                          q_.data() + h * head_dim, head_dim, cfg_.rms_norm_eps);
                    for (int h = 0; h < n_kv_heads; ++h)
                        backend_->rmsnorm(k_.data() + h * head_dim, w.k_norm,
                                          k_.data() + h * head_dim, head_dim, cfg_.rms_norm_eps);
                }

                // 2c. RoPE 旋转位置编码：把"位置 pos"的信息编进 q/k（v 不需要）
                {
                    ScopedTimer t(prof, scope("layer_%d.rope", i));
                    backend_->rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim, pos,
                         cfg_.rope_theta);
                }

                // 2d. 把当前 token 的 k/v 追加进 cache。注意必须先 append 再 attend：
                //     当前 token 要能"看到"自己，所以下面 attention 读的 seq_len = pos+1
                //     统一写入入口：内部按存储精度转换（fp32 memcpy / fp16 转换）
                {
                    ScopedTimer t(prof, scope("layer_%d.kv_append", i));
                    kv_.write_token(static_cast<int>(i), pos, k_.data(), v_.data());
                }

                // 2e. attention：当前 q 对 cache 里 [0..pos] 所有位置加权求和
                {
                    ScopedTimer t(prof, scope("layer_%d.attention", i));
                    attention_kv(q_.data(), static_cast<int>(i), pos + 1, n_heads, n_kv_heads,
                                 head_dim, attn_scale, attn_.data());
                }

                // 2f. 输出投影 o_proj
                {
                    ScopedTimer t(prof, scope("layer_%d.o_proj", i));
                    mv_rot(w.o_proj, attn_.data(), o_.data(), hidden, q_dim_, w.rot_o);
                }

                // 2g. 第一次残差连接：x = x + attention(x)。残差让梯度/信息能直通
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
                }
            }

            // ---- FFN 块（SwiGLU）：norm -> gate/up -> silu*up -> down ----
            // 三种架构的 FFN 完全一致，所以在 token mixer 分支之后共享

            // 2h. FFN 前的 RMSNorm
            {
                ScopedTimer t(prof, scope("layer_%d.post_attn_layernorm", i));
                backend_->rmsnorm(hidden_.data(), w.post_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }

            // 2i. FFN 块：dense SwiGLU 或 MoE（kQwen35MoE）二选一
            if (cfg_.is_moe()) {
                // ---- MoE FFN：路由门 -> topk softmax -> 共享专家 + Σ 路由专家 ----
                // 路由门（resident，[n_experts, hidden]）
                {
                    ScopedTimer t(prof, scope("layer_%d.moe_router", i));
                    mv(w.moe_router, normed_.data(), moe_gate_logits_.data(),
                       n_experts_, hidden);
                }
                // top-k 选择 + softmax 归一
                {
                    ScopedTimer t(prof, scope("layer_%d.topk_softmax", i));
                    backend_->topk_softmax(moe_gate_logits_.data(), n_experts_,
                                           experts_per_tok_, moe_topk_idx_.data(),
                                           moe_topk_w_.data());
                }
                // 共享专家（resident，权重 1）：gate/up -> swiglu -> down。
                // Qwen3-MoE 没有共享专家，此时 ffn_acc 从 0 起算，只累加路由专家。
                if (cfg_.has_shared_expert()) {
                    ScopedTimer t(prof, scope("layer_%d.shared_ffn", i));
                    mv_pair(w.moe_shared_gate, w.moe_shared_up, normed_.data(),
                            moe_expert_gate_.data(), moe_expert_up_.data(),
                            shared_inter_, hidden);
                    backend_->swiglu(moe_expert_gate_.data(), moe_expert_up_.data(),
                                     shared_inter_);
                    mv(w.moe_shared_down, moe_expert_gate_.data(),
                       moe_shared_out_.data(), hidden, shared_inter_);
                    for (int j = 0; j < hidden; ++j) moe_ffn_acc_[j] = moe_shared_out_[j];
                } else {
                    for (int j = 0; j < hidden; ++j) moe_ffn_acc_[j] = 0.0f;
                }

                // 路由专家：按 top-k 权重加权累加。权重走 resident 指针或
                // ExpertStore SSD 卸载（pread + LRU）——两者须逐位一致。
                for (int t = 0; t < experts_per_tok_; ++t) {
                    const int e = moe_topk_idx_[t];
                    const uint8_t *eg, *eu, *ed;
                    {
                        ScopedTimer tl(prof, scope("layer_%d.expert_load", i));
                        if (moe_ssd_) {
                            const ExpertWeights ew = expert_store_->get(
                                static_cast<int>(i), e);
                            eg = ew.gate; eu = ew.up; ed = ew.down;
                        } else {
                            eg = static_cast<const uint8_t *>(w.moe_experts[e].gate);
                            eu = static_cast<const uint8_t *>(w.moe_experts[e].up);
                            ed = static_cast<const uint8_t *>(w.moe_experts[e].down);
                        }
                    }
                    {
                        ScopedTimer tf(prof, scope("layer_%d.expert_ffn", i));
                        mv_pair(eg, eu, normed_.data(), moe_expert_gate_.data(),
                                moe_expert_up_.data(), moe_inter_, hidden);
                        backend_->swiglu(moe_expert_gate_.data(), moe_expert_up_.data(),
                                         moe_inter_);
                        mv(ed, moe_expert_gate_.data(), moe_expert_out_.data(),
                           hidden, moe_inter_);
                    }
                    const float wt = moe_topk_w_[t];
                    for (int j = 0; j < hidden; ++j)
                        moe_ffn_acc_[j] += wt * moe_expert_out_[j];
                }
                // 残差：x = x + moe_ffn
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_ffn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += moe_ffn_acc_[j];
                }
            } else {
                // ---- dense SwiGLU FFN ----
                // gate 和 up 两个投影
                if (rotated_) {
                    ScopedTimer t(prof, scope("layer_%d.gate_up_proj", i));
                    mv_rot(w.gate, normed_.data(), gate_.data(), inter, hidden, w.rot_gate);
                    mv_rot(w.up, normed_.data(), up_.data(), inter, hidden, w.rot_up);
                } else if (fuse_gate_up_) {
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
                // SwiGLU 融合
                {
                    ScopedTimer t(prof, scope("layer_%d.swiglu", i));
                    backend_->swiglu(gate_.data(), up_.data(), inter);
                }
                // down 投影
                {
                    ScopedTimer t(prof, scope("layer_%d.down_proj", i));
                    mv_rot(w.down, gate_.data(), ffn_.data(), hidden, inter, w.rot_down);
                }
                // 残差
                {
                    ScopedTimer t(prof, scope("layer_%d.residual_ffn", i));
                    for (int j = 0; j < hidden; ++j) hidden_[j] += ffn_[j];
                }
            }
        }

        // =====================================================================
        // Step 3 + 4: 最后的 norm + lm_head + argmax
        // =====================================================================
        // need_logits=false（prefill 非末位 token）时整段跳过：其 logits 会被
        // 丢弃，跳过省掉一次 lm_head 流量（4B ≈ 16%、0.8B ≈ 29% 单 token
        // 开销）。各层状态（KV cache / GDN state）在 Step 2 已照常更新，
        // 返回值 -1 仅被 forward_prefill 的回退循环用于中间迭代（随即被覆盖）。
        // 注意：--verbose / --dump-logits 的逐位置对照走 main 的直接调用，
        // 恒为 need_logits=true，对齐契约不受影响。
        int next = -1;
        if (need_logits) {
            {
                ScopedTimer t(prof, "final_norm");
                backend_->rmsnorm(hidden_.data(), final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
            }
            {
                // lm_head：把 hidden 向量投成 vocab 维的 logits（每个词一个分数）
                // tied 时 lm_head_ 就是 embed_（见 create）。I4 tied 时 embed 是 fp32
                ScopedTimer t(prof, "lm_head");
                if (lm_head_is_f32_) {
                    // I4 tied embeddings：lm_head 走 f32 matvec 路径
                    matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                               logits_.data(), vocab, hidden);
                } else if (lm_head_is_f16_) {
                    // VQ2 tied（embed 存 f16）：lm_head 走 f16 matvec
                    matvec_f16(static_cast<const uint16_t *>(lm_head_), normed_.data(),
                               logits_.data(), vocab, hidden);
                } else if (lm_head_is_i4_) {
                    // 紧凑 INT4 tied embed：lm_head 走 i4 matvec
                    matvec_i4(static_cast<const uint8_t *>(lm_head_), normed_.data(),
                              logits_.data(), vocab, hidden, group_size_);
                } else {
                    // 正常路径：通过 backend 的 matvec
                    mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
                }
            }

            // greedy 取 argmax：要 top-k 就顺便取，argmax 就是 top-1，不重复扫
            {
                ScopedTimer t(prof, "topk_argmax");
                if (topk) {
                    top_k_logits(logits_.data(), vocab, topk_k, topk);
                    next = topk->indices.empty() ? 0 : topk->indices[0]; // 第一个就是 argmax
                } else {
                    next = argmax(logits_.data(), vocab); // 直接取最大值索引
                }
            }
        }

        // 提交本 token 写入的 cache 槽位，收尾 profiler 记录
        kv_.advance(1);     // KV cache 长度 +1
        token_count_ += 1;  // 全局 token 计数 +1
        prof.end_token();   // 结束当前 token 的 profiling
        return next;
    }
} // namespace tinyqwen