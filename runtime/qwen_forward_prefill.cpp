// ============================================================================
// qwen_forward_prefill.cpp — Qwen2.x 批量 prefill 实现
// ============================================================================
// 本文件实现 QwenModel::forward_prefill() 方法，提供批量 prefill（GEMM 路径）
// 的高效实现。与逐 token 的 forward_token 不同，prefill 使用矩阵乘法（GEMM）
// 一次性处理所有 prompt token 的线性投影，显著降低延迟。
//
// 适用场景：非 verbose 模式下的标准 prefill 阶段。
// 限制：只支持 Qwen2.x（Qwen3.5 的 GDN linear attention 层需要顺序状态更新，
//       会回退到 forward_token 逐 token 处理）。
//
// 算法流程：
//   Step 1: 词嵌入 - 所有 token 的嵌入向量批量取出
//   Step 2: 逐层循环
//     2a. RMSNorm - 所有 token 的归一化
//     2b. Q/K/V 投影 - 通过 GEMM 批量计算
//     2c. Per-token 循环 - bias + RoPE + KV append + attention
//     2d. O 投影 - GEMM 批量计算
//     2e. 残差连接
//     2f. Post-attention RMSNorm
//     2g. Gate/Up 投影 - GEMM 批量计算
//     2h. SwiGLU - 逐 token 激活
//     2i. Down 投影 - GEMM 批量计算
//     2j. 残差连接
//   Step 3: 最终 norm + lm_head（仅最后一个 token）
//   Step 4: argmax（取下一个 token）
// ============================================================================

#include "qwen_model.h"

#include <algorithm>   // std::min, std::partial_sort
#include <cmath>       // std::sqrt
#include <cstdio>      // 标准输入输出（snprintf）
#include <cstring>     // 内存操作（memcpy）
#include <numeric>     // std::iota

#include "dispatch.h"  // matvec_f32, argmax
#include "ref_ops.h"   // ref 实现（作为兜底参考）

namespace tinyqwen {
    namespace {
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
    // QwenModel::forward_prefill() — 批量 prefill（GEMM 路径）
    // =========================================================================
    // 参数：
    //   token_ids — prompt token ID 数组
    //   n         — prompt token 数量
    //   topk      — 输出参数，不为 nullptr 时写入 top-k 结果
    //   topk_k    — top-k 的 k 值
    // 返回值：下一个要生成的 token ID
    // 说明：如果 n == 1，回退到 forward_token（单 token 不需要批处理）。
    //       如果 n == 0，返回 -1。
    //       如果模型是 Qwen3.5，回退到逐 token 的 forward_token（GDN 需要
    //       顺序状态更新）。
    //       批量 prefill 的算子结构与逐 token 不同，不做 op 级拆分——整批
    //       作为一条 prefill 记录计时。
    int QwenModel::forward_prefill(const int *token_ids, int n,
                                   TopKResult *topk, int topk_k) {
        if (n <= 0) return -1; // 空 prompt，直接返回
        if (n == 1) return forward_token(token_ids[0], topk, topk_k); // 单 token 回退

        // 提取配置参数（转为 int 便于循环中使用）
        const int hidden = static_cast<int>(cfg_.hidden_size);
        const int inter = static_cast<int>(cfg_.intermediate_size);
        const int vocab = static_cast<int>(cfg_.vocab_size);
        const int n_heads = static_cast<int>(cfg_.n_heads);
        const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
        const int head_dim = static_cast<int>(cfg_.head_dim);
        const int base_pos = kv_.seq_len(); // 当前 KV cache 中已存储的序列长度
        const bool is_qwen35 = cfg_.uses_qwen35_attention();

        // MoE 暂不支持批量 prefill GEMM 路径（专家 gather/scatter 未实现），
        // 一律强制逐 token（forward_token 已支持 MoE FFN）。这个分支必须在
        // 架构判断之前：dense-attention 的 kQwen3MoE 不属于 is_qwen35，否则会
        // 掉进下面 Qwen2 的批量路径——那条路径算的是 dense SwiGLU 而非 MoE，
        // 不报错但结果全错。
        if (cfg_.is_moe()) {
            // 批量路径优先：按专家分组，每专家只加载一次（省 ~50% I/O）
            if (batch_prefill_enabled_) {
                const int r = forward_prefill_moe_batch(token_ids, n, nullptr, 0);
                if (r != -2) return r;
                // fallthrough：逐 token 回退
            }
            int last = -1;
            for (int i = 0; i < n; ++i)
                last = forward_token(token_ids[i], (i == n - 1) ? topk : nullptr, topk_k,
                                     /*need_logits=*/i == n - 1);
            return last;
        }

        // GDN (linear attention) layers need sequential state updates — but the
        // 线性投影部分仍可批量。批量路径（qwen_forward_prefill_qwen35.cpp）：
        // 投影/FFN 走 GEMM（权重每层只读一遍），GDN 递归与因果 attention
        // 保留逐 token 顺序扫描。长 prompt 的 TTFT 大幅下降。
        // 无法处理时（无 BLAS 后端 / dtype 不支持 / 开关关闭 / token 太少）
        // 返回 -2，回退到逐 token。
        if (is_qwen35) {
            if (batch_prefill_enabled_ && n >= kBatchPrefillMinQwen35) {
                const int r = forward_prefill_qwen35_batch(token_ids, n, topk, topk_k);
                if (r != -2) return r;
                // fallthrough：逐 token 回退
            }
            int last = -1;
            for (int i = 0; i < n; ++i)
                last = forward_token(token_ids[i], (i == n - 1) ? topk : nullptr, topk_k,
                                     /*need_logits=*/i == n - 1);
            return last;
        }

        Profiler &prof = *profiler_;
        // attention 缩放因子 = 1 / sqrt(head_dim)，标准化点积的方差
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // 分配批量缓冲区（列主序存储：每列是一个 token 的向量）
        // 布局 [hidden, N]：第 c 列是第 c 个 token 的 hidden 向量
        const size_t N = static_cast<size_t>(n);
        std::vector<float> hid_batch(hidden * N);   // 残差流批量 [hidden, N]
        std::vector<float> norm_batch(hidden * N);  // RMSNorm 后批量 [hidden, N]
        std::vector<float> q_batch(q_dim_ * N);     // Q 投影批量 [q_dim, N]
        std::vector<float> k_batch(kv_dim_ * N);    // K 投影批量 [kv_dim, N]
        std::vector<float> v_batch(kv_dim_ * N);    // V 投影批量 [kv_dim, N]
        std::vector<float> attn_batch(q_dim_ * N);  // Attention 输出批量 [q_dim, N]
        std::vector<float> o_batch(hidden * N);     // o_proj 输出批量 [hidden, N]
        std::vector<float> gate_batch(inter * N);   // FFN gate 批量 [inter, N]
        std::vector<float> up_batch(inter * N);     // FFN up 批量 [inter, N]
        std::vector<float> ffn_batch(hidden * N);   // FFN 输出批量 [hidden, N]

        // =====================================================================
        // Step 1: 词嵌入 - 批量取出所有 token 的嵌入向量
        // =====================================================================
        {
            ScopedTimer t(prof, "prefill_embed");
            for (int c = 0; c < n; ++c) {
                int tid = token_ids[c]; // 当前 token 的 ID
                // 目标位置：hid_batch 的第 c 列（偏移 c * hidden）
                float *dst = hid_batch.data() + static_cast<size_t>(c) * hidden;
                // embed 卸载时按需 pread 单行（与 decode 侧同一套逻辑）
                const void *embed_src = embed_;
                size_t row_off = static_cast<size_t>(tid) * hidden;
                if (embed_file_offset_ != 0) {
                    // 行字节数随 dtype 变（fp32=hidden*4，fp16=hidden*2）；
                    // 硬编码 sizeof(float) 会让 fp16 embed 读到错误偏移的数据
                    const size_t row_bytes = static_cast<size_t>(hidden) *
                                             (embed_dtype_ == Dtype::kF16 ? 2 : 4);
                    if (!expert_store_ ||
                        !expert_store_->read_bytes(
                            embed_file_offset_ +
                                static_cast<uint64_t>(tid) * row_bytes,
                            row_bytes, embed_row_.data())) {
                        std::fprintf(stderr,
                                     "tinyqwen: embed 卸载 pread 失败 (token=%d)\n", tid);
                        std::abort();
                    }
                    embed_src = embed_row_.data();
                    row_off = 0;
                }
                if (embed_dtype_ == Dtype::kF32) {
                    // fp32 或 I4 文件（embed 存为 fp32 lookup table）：直接 memcpy
                    std::memcpy(dst, static_cast<const float *>(embed_src) + row_off,
                                hidden * sizeof(float));
                } else if (embed_dtype_ == Dtype::kI4) {
                    // 紧凑 INT4 embed：反量化该行
                    dequant_i4_row(static_cast<const uint8_t *>(embed_), tid, hidden,
                                   group_size_, dst);
                } else {
                    // f16 文件：逐元素转为 fp32
                    const uint16_t *row = static_cast<const uint16_t *>(embed_src) + row_off;
                    for (int j = 0; j < hidden; ++j) dst[j] = half_to_float(row[j]);
                }
            }
        }

        // =====================================================================
        // Step 2: 逐层循环（Transformer 层）
        // =====================================================================

        char name[64]; // profiler 作用域名的栈上缓冲区，避免每次分配 std::string
        for (uint32_t li = 0; li < cfg_.n_layers; ++li) {
            const LayerWeights &w = layers_[li];

            // 2a. RMSNorm：所有 token 的输入归一化（逐 token 独立）
            {
                std::snprintf(name, sizeof(name), "layer_%d.input_layernorm", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.input_ln,
                            norm_batch.data() + static_cast<size_t>(c) * hidden,
                            hidden, cfg_.rms_norm_eps);
                }
            }

            // 2b. Q/K/V 投影：通过 GEMM 批量计算（核心加速点）
            {
                std::snprintf(name, sizeof(name), "layer_%d.qkv_proj", li);
                ScopedTimer t(prof, name);
                // mm() 计算 y = W * x，其中 W 是 [M, K]，x 是 [K, N] 列主序
                mm_rot(w.q_proj, norm_batch.data(), q_batch.data(), q_dim_, hidden, n, w.rot_q);
                mm_rot(w.k_proj, norm_batch.data(), k_batch.data(), kv_dim_, hidden, n, w.rot_k);
                mm_rot(w.v_proj, norm_batch.data(), v_batch.data(), kv_dim_, hidden, n, w.rot_v);
            }

            // 2c. Per-token 循环：bias + RoPE + KV append + attention
            //     这部分无法批量化，因为 KV cache 写入和 attention 需要逐个位置更新
            {
                std::snprintf(name, sizeof(name), "layer_%d.attn_loop", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    const int pos = base_pos + c; // 当前 token 在序列中的绝对位置
                    // 第 c 列的各向量起始地址
                    float *qc = q_batch.data() + static_cast<size_t>(c) * q_dim_;
                    float *kc = k_batch.data() + static_cast<size_t>(c) * kv_dim_;
                    float *vc = v_batch.data() + static_cast<size_t>(c) * kv_dim_;
                    float *ac = attn_batch.data() + static_cast<size_t>(c) * q_dim_;

                    // bias 加法（Qwen2.x 特有，在 RoPE 之前）
                    if (w.q_bias) for (int j = 0; j < q_dim_; ++j) qc[j] += w.q_bias[j];
                    if (w.k_bias) for (int j = 0; j < kv_dim_; ++j) kc[j] += w.k_bias[j];
                    if (w.v_bias) for (int j = 0; j < kv_dim_; ++j) vc[j] += w.v_bias[j];

                    // QK per-head RMSNorm（Qwen3 稠密特有，同样在 RoPE 之前）
                    if (w.q_norm) {
                        for (int h = 0; h < n_heads; ++h)
                            backend_->rmsnorm(qc + h * head_dim, w.q_norm, qc + h * head_dim,
                                              head_dim, cfg_.rms_norm_eps);
                        for (int h = 0; h < n_kv_heads; ++h)
                            backend_->rmsnorm(kc + h * head_dim, w.k_norm, kc + h * head_dim,
                                              head_dim, cfg_.rms_norm_eps);
                    }

                    // RoPE 旋转位置编码：将位置 pos 的信息注入 Q 和 K
                    backend_->rope(qc, kc, n_heads, n_kv_heads, head_dim, pos, cfg_.rope_theta);

                    // KV append：将当前 token 的 K 和 V 写入 KV cache
                    // 统一写入入口：内部按存储精度转换（fp32 memcpy / fp16 转换）
                    kv_.write_token(static_cast<int>(li), pos, kc, vc);

                    // Attention：Q 对 KV cache 中 [0..pos] 所有位置加权求和
                    // seq_len = pos + 1（包含当前 token 自身）
                    // attention_kv 按 KV 精度分发（fp32 / fp16-KV 融合）
                    attention_kv(qc, static_cast<int>(li), pos + 1, n_heads, n_kv_heads,
                                 head_dim, attn_scale, ac);
                }
            }

            // 2d. O 投影：通过 GEMM 批量计算 + 残差连接
            {
                std::snprintf(name, sizeof(name), "layer_%d.o_proj", li);
                ScopedTimer t(prof, name);
                mm_rot(w.o_proj, attn_batch.data(), o_batch.data(), hidden, q_dim_, n, w.rot_o);
            }
            // 残差连接：x = x + attention(x)
            {
                std::snprintf(name, sizeof(name), "layer_%d.residual_attn", li);
                ScopedTimer t(prof, name);
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    hid_batch[j] += o_batch[j];
            }

            // 2e. Post-attention RMSNorm：所有 token 的 attention 后归一化
            {
                std::snprintf(name, sizeof(name), "layer_%d.post_attn_layernorm", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.post_ln,
                            norm_batch.data() + static_cast<size_t>(c) * hidden,
                            hidden, cfg_.rms_norm_eps);
                }
            }

            // 2f. FFN gate/up 投影：通过 GEMM 批量计算
            {
                std::snprintf(name, sizeof(name), "layer_%d.gate_up_proj", li);
                ScopedTimer t(prof, name);
                mm_rot(w.gate, norm_batch.data(), gate_batch.data(), inter, hidden, n, w.rot_gate);
                mm_rot(w.up, norm_batch.data(), up_batch.data(), inter, hidden, n, w.rot_up);
            }

            // 2g. SwiGLU 激活：逐 token 计算（gate 就地复用为融合结果）
            {
                std::snprintf(name, sizeof(name), "layer_%d.swiglu", li);
                ScopedTimer t(prof, name);
                for (int c = 0; c < n; ++c) {
                    backend_->swiglu(gate_batch.data() + static_cast<size_t>(c) * inter,
                           up_batch.data() + static_cast<size_t>(c) * inter, inter);
                }
            }

            // 2h. Down 投影：通过 GEMM 批量计算 + 残差连接
            {
                std::snprintf(name, sizeof(name), "layer_%d.down_proj", li);
                ScopedTimer t(prof, name);
                mm_rot(w.down, gate_batch.data(), ffn_batch.data(), hidden, inter, n, w.rot_down);
            }
            // 残差连接：x = x + ffn(x)
            {
                std::snprintf(name, sizeof(name), "layer_%d.residual_ffn", li);
                ScopedTimer t(prof, name);
                for (size_t j = 0; j < static_cast<size_t>(hidden) * N; ++j)
                    hid_batch[j] += ffn_batch[j];
            }
        }

        // =====================================================================
        // Step 3a（PPL 模式）: 全位置 norm + lm_head + 交叉熵
        // =====================================================================
        // ppl_mode_ 下不做生成，而是对每个位置 i 计算 logits 并对目标
        // token_ids[i+1] 累加 log_softmax（i = 0..n-2，共 n-1 个计分位置）。
        if (ppl_mode_) {
            ScopedTimer t(prof, "ppl_all_positions");
            for (int i = 0; i < n - 1; ++i) {
                const float *h = hid_batch.data() + static_cast<size_t>(i) * hidden;
                backend_->rmsnorm(h, final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
                if (lm_head_is_f32_) {
                    matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                               logits_.data(), vocab, hidden);
                } else if (lm_head_is_f16_) {
                    matvec_f16(static_cast<const uint16_t *>(lm_head_), normed_.data(),
                               logits_.data(), vocab, hidden);
                } else if (lm_head_is_i4_) {
                    matvec_i4(static_cast<const uint8_t *>(lm_head_), normed_.data(),
                              logits_.data(), vocab, hidden, group_size_);
                } else {
                    mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
                }
                ppl_sum_logprob_ += log_softmax_at(logits_.data(), vocab, token_ids[i + 1]);
                ++ppl_count_;
            }
            return 0; // PPL 模式不产出下一 token
        }

        // =====================================================================
        // Step 3: 最终 norm + lm_head（仅取最后一个 token 的 hidden state）
        // =====================================================================
        // 批量 prefill 只关心最后一个 token 的输出（即第一个生成 token）
        const float *last_hidden = hid_batch.data() + static_cast<size_t>(n - 1) * hidden;
        {
            ScopedTimer t(prof, "final_norm");
            backend_->rmsnorm(last_hidden, final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
        }
        // lm_head：将 hidden 向量投影到词表空间得到 logits
        {
            ScopedTimer t(prof, "lm_head");
            if (lm_head_is_f32_) {
                // I4 tied embeddings：embed 是 fp32，lm_head 走 f32 matvec
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

        // =====================================================================
        // Step 4: argmax — 取 logits 中分数最高的 token 作为下一个生成
        // =====================================================================
        int next = 0;
        {
            ScopedTimer t(prof, "topk_argmax");
            if (topk) {
                top_k_logits(logits_.data(), vocab, topk_k, topk);
                next = topk->indices.empty() ? 0 : topk->indices[0]; // 第一个就是 argmax
            } else {
                next = argmax(logits_.data(), vocab); // 直接取最大值索引
            }
        }

        // 推进 KV cache 和 token 计数器（所有 N 个 token 都已写入 cache）
        kv_.advance(n);
        token_count_ += n;
        return next;
    }
    // =========================================================================
    // QwenModel::forward_ppl() — 批量 prefill + 全位置交叉熵（困惑度）
    // =========================================================================
    // 置 ppl_mode_ 后复用 forward_prefill 的层计算；其 Step 3a 分支完成
    // 全位置 norm + lm_head + log_softmax 累加。返回平均 NLL。
    double QwenModel::forward_ppl(const int *token_ids, int n, long *out_count) {
        ppl_mode_ = true;
        ppl_sum_logprob_ = 0.0;
        ppl_count_ = 0;
        if (cfg_.model_type == ModelType::kQwen35) {
            forward_prefill_qwen35_batch(token_ids, n, nullptr, 0); // Qwen3.5 批量路径
        } else {
            forward_prefill(token_ids, n, nullptr, 0);              // Qwen2.x
        }
        ppl_mode_ = false;
        if (out_count) *out_count = ppl_count_;
        return ppl_count_ > 0 ? -ppl_sum_logprob_ / static_cast<double>(ppl_count_) : 0.0;
    }

    // =====================================================================
    // QwenModel::forward_prefill_moe_batch() — MoE 批量 prefill
    // =====================================================================
    // 逐 token prefill 的浪费：同一专家被多个 token 重复选中，却每次都重新读盘、
    // 各算一次 matvec。实测 n=32 时 bytes_read 57.4 GB、hits=0（零复用）。
    //
    // 本路径改为**按专家分组**：router 算完全部 N token → 按专家归组 → 每专家
    // 只加载一次 → 对其所有 token 做一次 batch GEMM → 按 topk 权重散回各 token。
    //
    // 批量的部分：qkv_proj / o_proj / router / 专家 FFN（都走 matmul_gptq）。
    // 顺序的部分：rope / kv_append / causal attention（token i 只看 0..i，
    // 必须顺序扫描；但 attention 只占 0.5%，不影响收益）。
    //
    // 返回最后一个 token 的 argmax；无法处理时返回 -2 让调用方回退逐 token。
    int QwenModel::forward_prefill_moe_batch(const int *token_ids, int n,
                                             int *topk, int topk_k) {
        if (!cfg_.is_moe() || n <= 0) return -2;
        // 专家权重必须是 GPTQ（matmul_gptq 只支持该布局）
        if (expert_dtype_ != Dtype::kGPTQ4) return -2;
        // 混合架构（GDN + full attention）这条路没有 GDN 算子实现——层循环只算
        // full attention。Qwen3.5-MoE 是 GDN:full = 3:1 混合，走这里会把线性层
        // 当 full attention 算，架构就错了（不报错但结果全错）。回退逐 token
        // 路径（那条有完整 GDN 实现）。必须在写 KV 之前判断，否则回退会踩脏 KV。
        if (cfg_.full_attention_interval > 1) return -2;
        // attention / o_proj / 共享专家也必须是 GPTQ 布局：真 checkpoint 里它们
        // 常是 bf16/fp16，matmul_gptq 会把 fp16 当 GPTQ in-band 块解析 →
        // 段错误或垃圾（AGENTS.md 坑 #19 同类）。
        {
            const LayerWeights &w0 = layers_[0];
            if (w0.attn_dtype != Dtype::kGPTQ4) return -2;
            if (cfg_.has_shared_expert() && w0.shared_dtype != Dtype::kGPTQ4) return -2;
        }
        // 小 n 时批处理的固定开销（缓冲分配、每列重算 sx8）超过专家复用收益：
        // 实测 n=8 批处理 0.97×（略慢），n=32 才 1.21×，n=128 达 1.44×。
        // 复用收益随 n 增长（每层 n×8 次选择从 128 个专家里挑，重叠率随 n 升）。
        constexpr int kMoEBatchPrefillMinN = 16;
        if (n < kMoEBatchPrefillMinN) return -2;

        Profiler &prof = *profiler_;
        const int hidden = cfg_.hidden_size;
        const int head_dim = static_cast<int>(cfg_.head_dim);
        const int N = n;
        const size_t hN = static_cast<size_t>(hidden) * N;
        char name[64];
        const auto scope = [&](const char *fmt, int layer) {
            std::snprintf(name, sizeof(name), fmt, layer);
            return name;
        };

        // 缓冲（prefill 只跑一次，分配开销可忽略）
        std::vector<float> hid(hN), normed(hN), out(hN);
        std::vector<float> qkv_buf(static_cast<size_t>(q_dim_ + 2 * kv_dim_) * N);
        std::vector<float> attn_buf(static_cast<size_t>(q_dim_) * N);
        std::vector<float> gate_logits(static_cast<size_t>(n_experts_) * N);
        std::vector<float> ffn_acc(hN, 0.0f);
        std::vector<int> topk_idx(static_cast<size_t>(experts_per_tok_) * N);
        std::vector<float> topk_w(static_cast<size_t>(experts_per_tok_) * N);
        // 专家分组用
        std::vector<std::vector<int>> groups(static_cast<size_t>(n_experts_));
        std::vector<float> X_g, gate_g, up_g, down_g;

        // ---- Step 1: 词嵌入 ----
        {
            ScopedTimer t(prof, "prefill_embed");
            for (int c = 0; c < N; ++c) {
                const int tid = token_ids[c];
                float *dst = hid.data() + static_cast<size_t>(c) * hidden;
                if (embed_file_offset_ != 0) {
                    const size_t row_bytes = static_cast<size_t>(hidden) *
                                             (embed_dtype_ == Dtype::kF16 ? 2 : 4);
                    if (!expert_store_ ||
                        !expert_store_->read_bytes(
                            embed_file_offset_ + static_cast<uint64_t>(tid) * row_bytes,
                            row_bytes, embed_row_.data())) {
                        std::fprintf(stderr,
                                     "tinyqwen: embed 卸载 pread 失败 (token=%d)\n", tid);
                        std::abort();
                    }
                    if (embed_dtype_ == Dtype::kF16) {
                        const uint16_t *row =
                            reinterpret_cast<const uint16_t *>(embed_row_.data());
                        for (int j = 0; j < hidden; ++j) dst[j] = half_to_float(row[j]);
                    } else {
                        std::memcpy(dst, embed_row_.data(), hidden * sizeof(float));
                    }
                } else if (embed_dtype_ == Dtype::kF32) {
                    std::memcpy(dst, static_cast<const float *>(embed_) +
                                         static_cast<size_t>(tid) * hidden,
                                hidden * sizeof(float));
                } else if (embed_dtype_ == Dtype::kF16) {
                    const uint16_t *row = static_cast<const uint16_t *>(embed_) +
                                          static_cast<size_t>(tid) * hidden;
                    for (int j = 0; j < hidden; ++j) dst[j] = half_to_float(row[j]);
                } else {
                    return -2;   // i4 embed 暂不支持批量路径
                }
            }
        }

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // ---- Step 2: 逐层 ----
        for (int i = 0; i < cfg_.n_layers; ++i) {
            const LayerWeights &w = layers_[static_cast<size_t>(i)];

            // 2a. input_layernorm（逐 token）
            {
                ScopedTimer t(prof, scope("layer_%d.input_layernorm", i));
                for (int c = 0; c < N; ++c) {
                    backend_->rmsnorm(hid.data() + static_cast<size_t>(c) * hidden,
                                      w.input_ln,
                                      normed.data() + static_cast<size_t>(c) * hidden,
                                      hidden, cfg_.rms_norm_eps);
                }
            }

            // 2b. qkv_proj：批量 GEMM（权重每层只读一遍）
            {
                ScopedTimer t(prof, scope("layer_%d.qkv_proj", i));
                const int qkv_out = q_dim_ + 2 * kv_dim_;
                // 三次独立 GEMM 写进同一缓冲的不同区段（列主序：每投影占 N 列）
                matmul_gptq(static_cast<const uint8_t *>(w.q_proj), normed.data(),
                            qkv_buf.data(), q_dim_, hidden, N, gptq_group_size_);
                matmul_gptq(static_cast<const uint8_t *>(w.k_proj), normed.data(),
                            qkv_buf.data() + static_cast<size_t>(q_dim_) * N,
                            kv_dim_, hidden, N, gptq_group_size_);
                matmul_gptq(static_cast<const uint8_t *>(w.v_proj), normed.data(),
                            qkv_buf.data() + static_cast<size_t>(q_dim_ + kv_dim_) * N,
                            kv_dim_, hidden, N, gptq_group_size_);
                (void)qkv_out;
            }

            // 2c. rope + kv_append + causal attention（顺序扫描，token i 只看 0..i）
            {
                ScopedTimer t(prof, scope("layer_%d.attention", i));
                for (int c = 0; c < N; ++c) {
                    float *qc = qkv_buf.data() + static_cast<size_t>(c) * q_dim_;
                    float *kc = qkv_buf.data() + static_cast<size_t>(N) * q_dim_ +
                                static_cast<size_t>(c) * kv_dim_;
                    float *vc = qkv_buf.data() + static_cast<size_t>(N) * (q_dim_ + kv_dim_) +
                                static_cast<size_t>(c) * kv_dim_;
                    if (w.q_norm) {
                        for (int h = 0; h < cfg_.n_heads; ++h)
                            backend_->rmsnorm(qc + h * head_dim, w.q_norm,
                                              qc + h * head_dim, head_dim,
                                              cfg_.rms_norm_eps);
                        for (int h = 0; h < cfg_.n_kv_heads; ++h)
                            backend_->rmsnorm(kc + h * head_dim, w.k_norm,
                                              kc + h * head_dim, head_dim,
                                              cfg_.rms_norm_eps);
                    }
                    backend_->rope(qc, kc, cfg_.n_heads, cfg_.n_kv_heads, head_dim,
                                   c, cfg_.rope_theta);
                    // 必须先 write_token 再 attend：当前 token 要能看到自己
                    kv_.write_token(i, c, kc, vc);
                    attention_kv(qc, i, c + 1, cfg_.n_heads, cfg_.n_kv_heads,
                                 head_dim, attn_scale,
                                 attn_buf.data() + static_cast<size_t>(c) * q_dim_);
                }
            }

            // 2d. o_proj：批量 GEMM + 残差
            {
                ScopedTimer t(prof, scope("layer_%d.o_proj", i));
                matmul_gptq(static_cast<const uint8_t *>(w.o_proj), attn_buf.data(),
                            out.data(), hidden, q_dim_, N, gptq_group_size_);
                for (size_t j = 0; j < hN; ++j) hid[j] += out[j];
            }

            // 2e. post_attn_layernorm（逐 token）
            {
                ScopedTimer t(prof, scope("layer_%d.post_attn_layernorm", i));
                for (int c = 0; c < N; ++c) {
                    backend_->rmsnorm(hid.data() + static_cast<size_t>(c) * hidden,
                                      w.post_ln,
                                      normed.data() + static_cast<size_t>(c) * hidden,
                                      hidden, cfg_.rms_norm_eps);
                }
            }

            // 2f. router：批量 GEMM → gate_logits [n_experts, N]
            {
                ScopedTimer t(prof, scope("layer_%d.moe_router", i));
                if (moe_router_dtype_ == Dtype::kGPTQ4) {
                    matmul_gptq(static_cast<const uint8_t *>(w.moe_router), normed.data(),
                                gate_logits.data(), n_experts_, hidden, N,
                                gptq_group_size_);
                } else {
                    // router 按**自身 dtype** 逐 token matvec（router 很小，占比 0.1%）。
                    // 真 checkpoint 的 router 是 fp16——按 fp32 读会把 fp16 字节当
                    // fp32 解析，得到垃圾并产生 NaN（AGENTS.md 坑 #24 同类）。
                    for (int c = 0; c < N; ++c) {
                        float *dst = gate_logits.data() +
                                     static_cast<size_t>(c) * n_experts_;
                        const float *src = normed.data() + static_cast<size_t>(c) * hidden;
                        if (moe_router_dtype_ == Dtype::kF16) {
                            matvec_f16(static_cast<const uint16_t *>(w.moe_router),
                                       src, dst, n_experts_, hidden);
                        } else {
                            matvec_f32(static_cast<const float *>(w.moe_router),
                                       src, dst, n_experts_, hidden);
                        }
                    }
                }
            }

            // 2g. topk（逐 token）+ 按专家分组
            {
                ScopedTimer t(prof, scope("layer_%d.topk_softmax", i));
                for (size_t e = 0; e < groups.size(); ++e) groups[e].clear();
                for (int c = 0; c < N; ++c) {
                    int *idx = topk_idx.data() + static_cast<size_t>(c) * experts_per_tok_;
                    float *wt = topk_w.data() + static_cast<size_t>(c) * experts_per_tok_;
                    backend_->topk_softmax(
                        gate_logits.data() + static_cast<size_t>(c) * n_experts_,
                        n_experts_, experts_per_tok_, idx, wt);
                    for (int t = 0; t < experts_per_tok_; ++t) {
                        groups[static_cast<size_t>(idx[t])].push_back(c);
                    }
                }
            }

            // 2h. 共享专家（若有）：批量 GEMM
            if (cfg_.has_shared_expert()) {
                ScopedTimer t(prof, scope("layer_%d.shared_ffn", i));
                std::vector<float> sg(static_cast<size_t>(shared_inter_) * N);
                std::vector<float> su(static_cast<size_t>(shared_inter_) * N);
                matmul_gptq(static_cast<const uint8_t *>(w.moe_shared_gate), normed.data(),
                            sg.data(), shared_inter_, hidden, N, gptq_group_size_);
                matmul_gptq(static_cast<const uint8_t *>(w.moe_shared_up), normed.data(),
                            su.data(), shared_inter_, hidden, N, gptq_group_size_);
                for (int c = 0; c < N; ++c) {
                    backend_->swiglu(sg.data() + static_cast<size_t>(c) * shared_inter_,
                                     su.data() + static_cast<size_t>(c) * shared_inter_,
                                     shared_inter_);
                }
                matmul_gptq(static_cast<const uint8_t *>(w.moe_shared_down), sg.data(),
                            ffn_acc.data(), hidden, shared_inter_, N, gptq_group_size_);
            } else {
                for (size_t j = 0; j < hN; ++j) ffn_acc[j] = 0.0f;
            }

            // 2i. 路由专家：**按专家分组，每专家只加载一次 + 一次 batch GEMM**
            {
                ScopedTimer t(prof, scope("layer_%d.expert_ffn", i));
                for (int e = 0; e < n_experts_; ++e) {
                    const std::vector<int> &toks = groups[static_cast<size_t>(e)];
                    if (toks.empty()) continue;      // 本层没有 token 选中该专家
                    const int M = static_cast<int>(toks.size());

                    const uint8_t *eg, *eu, *ed;
                    if (moe_ssd_) {
                        const ExpertWeights ew =
                            expert_store_->get(i, e);   // 整层只读一次
                        eg = ew.gate; eu = ew.up; ed = ew.down;
                    } else {
                        eg = static_cast<const uint8_t *>(w.moe_experts[e].gate);
                        eu = static_cast<const uint8_t *>(w.moe_experts[e].up);
                        ed = static_cast<const uint8_t *>(w.moe_experts[e].down);
                    }

                    // gather：把该专家的 token 的 normed 向量拼成 X_g [hidden, M]
                    X_g.resize(static_cast<size_t>(hidden) * M);
                    for (int m = 0; m < M; ++m) {
                        std::memcpy(X_g.data() + static_cast<size_t>(m) * hidden,
                                    normed.data() +
                                        static_cast<size_t>(toks[m]) * hidden,
                                    hidden * sizeof(float));
                    }

                    gate_g.resize(static_cast<size_t>(moe_inter_) * M);
                    up_g.resize(static_cast<size_t>(moe_inter_) * M);
                    matmul_gptq(eg, X_g.data(), gate_g.data(), moe_inter_, hidden, M,
                                gptq_group_size_);
                    matmul_gptq(eu, X_g.data(), up_g.data(), moe_inter_, hidden, M,
                                gptq_group_size_);
                    // 列主序下每列天然连续，swiglu 可直接按列调用
                    for (int m = 0; m < M; ++m) {
                        backend_->swiglu(gate_g.data() + static_cast<size_t>(m) * moe_inter_,
                                         up_g.data() + static_cast<size_t>(m) * moe_inter_,
                                         moe_inter_);
                    }
                    down_g.resize(hN > 0 ? static_cast<size_t>(hidden) * M : 0);
                    matmul_gptq(ed, gate_g.data(), down_g.data(), hidden, moe_inter_, M,
                                gptq_group_size_);

                    // scatter：按 topk 权重累加回各 token
                    for (int m = 0; m < M; ++m) {
                        const int c = toks[m];
                        // 找到该 token 选中专家 e 的位置以取权重
                        const int *idx = topk_idx.data() +
                                         static_cast<size_t>(c) * experts_per_tok_;
                        const float *wt = topk_w.data() +
                                          static_cast<size_t>(c) * experts_per_tok_;
                        float w_e = 0.0f;
                        for (int t = 0; t < experts_per_tok_; ++t) {
                            if (idx[t] == e) { w_e = wt[t]; break; }
                        }
                        float *acc = ffn_acc.data() + static_cast<size_t>(c) * hidden;
                        const float *src = down_g.data() + static_cast<size_t>(m) * hidden;
                        for (int j = 0; j < hidden; ++j) acc[j] += w_e * src[j];
                    }
                }
            }

            // 2j. 残差
            {
                ScopedTimer t(prof, scope("layer_%d.residual_ffn", i));
                for (size_t j = 0; j < hN; ++j) hid[j] += ffn_acc[j];
            }
        }

        // ---- Step 3: final norm + lm_head（只算最后一列）----
        int last = -1;
        {
            ScopedTimer t(prof, "final_norm");
            backend_->rmsnorm(hid.data() + static_cast<size_t>(N - 1) * hidden,
                              final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
        }
        {
            ScopedTimer t(prof, "lm_head");
            if (lm_head_dtype_ == Dtype::kGPTQ4) {
                matvec_gptq(static_cast<const uint8_t *>(lm_head_), normed_.data(),
                            logits_.data(), cfg_.vocab_size, hidden, gptq_group_size_);
            } else if (lm_head_dtype_ == Dtype::kF16) {
                matvec_f16(static_cast<const uint16_t *>(lm_head_), normed_.data(),
                           logits_.data(), cfg_.vocab_size, hidden);
            } else {
                matvec_f32(static_cast<const float *>(lm_head_), normed_.data(),
                           logits_.data(), cfg_.vocab_size, hidden);
            }
            last = backend_->argmax(logits_.data(), cfg_.vocab_size);
        }
        kv_.advance(N);
        (void)topk; (void)topk_k;   // 批量路径不输出 topk（prefill 只需末位 argmax）
        return last;
    }

} // namespace tinyqwen