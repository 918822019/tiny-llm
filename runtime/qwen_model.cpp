// Qwen 前向的实现。这是整个 runtime 的核心，建议配合 docs/qwen_forward.md 阅读。
//
// 一个 token 的前向大致是：
//   查词嵌入 -> 逐层 [attention + FFN] -> 最后 norm -> 投影到词表得 logits -> argmax
//
// 刻意不做任何图抽象：forward 就是一个可读的函数，op 顺序与文档一一对应，
// 跟 PyTorch 出现数值偏差时肉眼可查。
//
// workspace buffer（create() 时一次分配，每个 token 不再分配）：
//   hidden_  残差流：贯穿所有层的主数据，残差不断往上加
//   normed_  每次 RMSNorm 的输出，喂给接下来的投影
//   q_/k_/v_ attention 的 query/key/value 投影结果
//   attn_    attention 输出（各 head 拼接）
//   o_       o_proj 输出
//   gate_/up_/ffn_  SwiGLU FFN 的中间量
//   logits_  lm_head 输出，对词表每个词一个分数

#include "qwen_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "dispatch.h"  // matvec_f32 通用入口（分发到 _ref / 将来的优化版）
#include "gdn_ops.h"   // Qwen3.5 GDN 算子（l2norm / conv1d / delta rule / ...）
#include "ref_ops.h"

namespace tinyqwen {
    namespace {
        // HF GDN 的门控标量运算（fp32）：
        //   beta = sigmoid(b)；g = -exp(A_log) * softplus(a + dt_bias)。
        // softplus 与 PyTorch F.softplus 一致（threshold=20：大输入直接返回 x，
        // 避免 exp 溢出）。
        inline float sigmoidf32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

        inline float softplusf32(float x) {
            return x > 20.0f ? x : std::log1p(std::exp(x));
        }
        // 取分数最高的 k 个 token。实现：对下标数组做 partial_sort，
        // 只把前 k 个排好序，复杂度 O(vocab * k)。结果第一个就是 argmax。
        // v1 够用（每 token 调一次，不在 kernel 热点路径上）。
        void top_k_logits(const float *logits, int vocab, int k, TopKResult *out) {
            k = std::min(k, vocab);
            std::vector<int> idx(vocab);
            std::iota(idx.begin(), idx.end(), 0); // idx = [0,1,2,...]
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [logits](int a, int b) { return logits[a] > logits[b]; });
            out->indices.resize(k);
            out->values.resize(k);
            for (int i = 0; i < k; ++i) {
                out->indices[i] = idx[i];
                out->values[i] = logits[idx[i]];
            }
        }

        std::string shape_str(const std::vector<uint64_t> &s) {
            std::string r = "[";
            for (size_t i = 0; i < s.size(); ++i) {
                if (i) r += ", ";
                r += std::to_string(s[i]);
            }
            return r + "]";
        }
    } // namespace

    // 按 HF 名字取 tensor，并校验 ndim/shape/dtype 是否与 forward 预期一致。
    // shape 不对（比如导出了非 Qwen 的 checkpoint）在这里就变成明确报错，
    // 而不是后面悄悄算错（fail fast，见 primer 第 9 节）。
    // dtype 必须等于文件级 dtype（v1 单一 dtype，loader 已保证；这里双保险）。
    const TensorView *QwenModel::require_view(const ModelFile &file, const std::string &name,
                                              const std::vector<uint64_t> &shape,
                                              std::string *err) {
        const TensorView *t = file.get(name);
        if (!t) {
            if (err) *err = "missing tensor: " + name;
            return nullptr;
        }
        // I4 文件允许混合 dtype：大矩阵 kI4、小向量 kF32。
        if (dtype_ == Dtype::kI4) {
            if (t->dtype != Dtype::kI4 && t->dtype != Dtype::kF32) {
                if (err)
                    *err = "tensor " + name + " dtype mismatch: i4 file allows i4/f32, got " +
                           dtype_name(t->dtype);
                return nullptr;
            }
        } else if (t->dtype != dtype_) {
            if (err)
                *err = "tensor " + name + " dtype mismatch: got " + dtype_name(t->dtype) +
                       " expected " + dtype_name(dtype_);
            return nullptr;
        }
        if (static_cast<size_t>(t->ndim) != shape.size()) {
            if (err)
                *err = "tensor " + name + " ndim mismatch: got " + std::to_string(t->ndim) +
                       " expected " + std::to_string(shape.size());
            return nullptr;
        }
        for (size_t i = 0; i < shape.size(); ++i) {
            if (t->shape[i] != shape[i]) {
                if (err) {
                    const std::vector<uint64_t> got_shape(t->shape, t->shape + t->ndim);
                    *err = "tensor " + name + " shape mismatch: got " + shape_str(got_shape) +
                           " expected " + shape_str(shape);
                }
                return nullptr;
            }
        }
        return t;
    }

    // norm/bias 这类小向量恒按 fp32 使用：f32 模型零拷贝直指文件内存；
    // f16 模型在 create 期一次性转成 fp32 副本（总量 ~400KB 级，换算耗时
    // 可忽略），换来 rmsnorm/bias 的每 token 路径不需要感知 dtype。
    const float *QwenModel::bind_f32_vector(const TensorView *t) {
        // I4 文件中小向量存为 kF32，与纯 f32 文件相同：零拷贝直指。
        if (dtype_ == Dtype::kF32 || t->dtype == Dtype::kF32) return t->f32();
        // f16 模型：逐元素转换成 fp32 副本。
        std::vector<float> buf(t->numel());
        const uint8_t *src = t->data;
        for (size_t i = 0; i < buf.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, src + i * 2, sizeof(h));
            buf[i] = half_to_float(h);
        }
        owned_f32_.push_back(std::move(buf));
        return owned_f32_.back().data();
    }

    // matvec 分派薄封装：kernel 侧的通用入口由 dispatch 继续分发到具体变体。
    void QwenModel::mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const {
        if (dtype_ == Dtype::kI4) {
            matvec_i4(static_cast<const uint8_t *>(w), x, y, out_dim, in_dim, group_size_);
        } else if (dtype_ == Dtype::kF16) {
            matvec_f16(static_cast<const uint16_t *>(w), x, y, out_dim, in_dim);
        } else {
            matvec_f32(static_cast<const float *>(w), x, y, out_dim, in_dim);
        }
    }

    void QwenModel::mv_pair(const void *w1, const void *w2, const float *x, float *y1,
                            float *y2, int out_dim, int in_dim) const {
        if (dtype_ == Dtype::kI4) {
            matvec_pair_i4(static_cast<const uint8_t *>(w1),
                           static_cast<const uint8_t *>(w2), x, y1, y2, out_dim, in_dim,
                           group_size_);
        } else if (dtype_ == Dtype::kF16) {
            matvec_pair_f16(static_cast<const uint16_t *>(w1),
                            static_cast<const uint16_t *>(w2), x, y1, y2, out_dim, in_dim);
        } else {
            matvec_pair_f32(static_cast<const float *>(w1), static_cast<const float *>(w2), x,
                            y1, y2, out_dim, in_dim);
        }
    }

    void QwenModel::mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                           float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                           int in_dim) const {
        if (dtype_ == Dtype::kI4) {
            matvec_qkv_i4(static_cast<const uint8_t *>(wq),
                          static_cast<const uint8_t *>(wk),
                          static_cast<const uint8_t *>(wv), x, yq, yk, yv,
                          q_dim, kv_dim, in_dim, group_size_);
        } else if (dtype_ == Dtype::kF16) {
            matvec_qkv_f16(static_cast<const uint16_t *>(wq),
                           static_cast<const uint16_t *>(wk),
                           static_cast<const uint16_t *>(wv), x, yq, yk, yv,
                           q_dim, kv_dim, in_dim);
        } else {
            matvec_qkv_f32(static_cast<const float *>(wq), static_cast<const float *>(wk),
                           static_cast<const float *>(wv), x, yq, yk, yv,
                           q_dim, kv_dim, in_dim);
        }
    }

    // 工厂：绑定权重视图、校验每个 tensor、初始化 KV cache 和 workspace。
    // 所有失败经 *err 报告；成功后 *out 持有一个可直接运行的模型。
    bool QwenModel::create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out) {
        out->reset();
        if (!file.loaded()) {
            if (err) *err = "model file not loaded";
            return false;
        }
        const ModelConfig &cfg = file.config();

        // max_seq_len <= 0 表示"用模型训练时的上限"。
        if (max_seq_len <= 0) max_seq_len = static_cast<int>(cfg.max_seq_len);
        if (max_seq_len > static_cast<int>(cfg.max_seq_len)) {
            if (err)
                *err = "max_seq_len " + std::to_string(max_seq_len) +
                       " exceeds model max " + std::to_string(cfg.max_seq_len);
            return false;
        }

        std::unique_ptr<QwenModel> m(new QwenModel());
        m->cfg_ = cfg;
        m->dtype_ = static_cast<Dtype>(file.header().dtype);
        m->group_size_ = static_cast<int>(cfg.quant_group_size);
        m->profiler_ = &profiler;
        m->max_seq_len_ = max_seq_len;
        m->q_dim_ = static_cast<int>(cfg.n_heads * cfg.head_dim);
        m->kv_dim_ = static_cast<int>(cfg.n_kv_heads * cfg.head_dim);

        const bool is_qwen35 = cfg.model_type == ModelType::kQwen35;
        if (is_qwen35) {
            // GDN / partial RoPE 的派生维度（forward 里反复用，先算好）。
            m->gdn_qk_dim_ = static_cast<int>(cfg.linear_num_qk_heads * cfg.linear_qk_head_dim);
            m->gdn_value_dim_ =
                    static_cast<int>(cfg.linear_num_v_heads * cfg.linear_v_head_dim);
            m->gdn_conv_dim_ = 2 * m->gdn_qk_dim_ + m->gdn_value_dim_;
            m->rotary_dim_ =
                    static_cast<int>(std::lround(cfg.head_dim * cfg.partial_rotary_factor));
            if (m->rotary_dim_ % 2 != 0 || m->rotary_dim_ > static_cast<int>(cfg.head_dim)) {
                if (err)
                    *err = "partial_rotary_factor yields odd/oversized rotary_dim: " +
                           std::to_string(m->rotary_dim_);
                return false;
            }
        }

        const uint64_t hidden = cfg.hidden_size;
        const uint64_t inter = cfg.intermediate_size;
        const uint64_t vocab = cfg.vocab_size;
        const uint64_t qd = static_cast<uint64_t>(m->q_dim_);
        const uint64_t kvd = static_cast<uint64_t>(m->kv_dim_);

        // 绑定辅助：大矩阵取裸字节指针（dtype 随模型），小向量恒绑 fp32。
        const auto bind_mat = [&](const char *name, std::vector<uint64_t> shape,
                                  const void **out) -> bool {
            const TensorView *t = m->require_view(file, name, shape, err);
            if (!t) return false;
            *out = t->data;
            return true;
        };
        const auto bind_vec = [&](const char *name, std::vector<uint64_t> shape,
                                  const float **out) -> bool {
            const TensorView *t = m->require_view(file, name, shape, err);
            if (!t) return false;
            *out = m->bind_f32_vector(t);
            return true;
        };

        // 全局权重：词嵌入、最后 norm、lm_head。
        if (!bind_mat("model.embed_tokens.weight", {vocab, hidden}, &m->embed_)) return false;
        if (!bind_vec("model.norm.weight", {hidden}, &m->final_norm_)) return false;
        if (cfg.tied_embeddings) {
            // tied：lm_head 和词嵌入共享同一份权重（0.5B 模型如此），省一份内存。
            m->lm_head_ = m->embed_;
            // I4 文件中 embed 存为 fp32（lookup table 不量化），lm_head 投影需走 f32 路径。
            if (m->dtype_ == Dtype::kI4) m->lm_head_is_f32_ = true;
        } else {
            if (!bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) return false;
        }

        // 每层权重。tensor 名字沿用 HuggingFace 约定（qwen35 的 linear_attn
        // 子模块名同样照搬），缺哪个名字就能直接定位到 exporter 的问题。
        const uint64_t gdn_v = static_cast<uint64_t>(m->gdn_value_dim_);
        const uint64_t gdn_conv = static_cast<uint64_t>(m->gdn_conv_dim_);
        const uint64_t n_v_heads = cfg.linear_num_v_heads;
        const uint64_t v_head_dim = cfg.linear_v_head_dim;
        const uint64_t conv_k = cfg.linear_conv_kernel_dim;

        m->layers_.resize(cfg.n_layers);
        for (uint32_t i = 0; i < cfg.n_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            LayerWeights &w = m->layers_[i];
            // norm / FFN 两种架构完全一致（SwiGLU，无 bias）。
            if (!bind_vec((p + "input_layernorm.weight").c_str(), {hidden}, &w.input_ln))
                return false;
            if (!bind_vec((p + "post_attention_layernorm.weight").c_str(), {hidden}, &w.post_ln))
                return false;
            if (!bind_mat((p + "mlp.gate_proj.weight").c_str(), {inter, hidden}, &w.gate))
                return false;
            if (!bind_mat((p + "mlp.up_proj.weight").c_str(), {inter, hidden}, &w.up))
                return false;
            if (!bind_mat((p + "mlp.down_proj.weight").c_str(), {hidden, inter}, &w.down))
                return false;

            if (is_qwen35 && cfg.is_linear_layer(i)) {
                // ---- Gated DeltaNet 层 ----
                const std::string g = p + "linear_attn.";
                if (!bind_mat((g + "in_proj_qkv.weight").c_str(), {gdn_conv, hidden},
                              &w.gdn_in_qkv)) return false;
                if (!bind_mat((g + "in_proj_z.weight").c_str(), {gdn_v, hidden}, &w.gdn_in_z))
                    return false;
                if (!bind_mat((g + "in_proj_b.weight").c_str(), {n_v_heads, hidden},
                              &w.gdn_in_b)) return false;
                if (!bind_mat((g + "in_proj_a.weight").c_str(), {n_v_heads, hidden},
                              &w.gdn_in_a)) return false;
                if (!bind_mat((g + "out_proj.weight").c_str(), {hidden, gdn_v}, &w.gdn_out_proj))
                    return false;
                if (!bind_vec((g + "conv1d.weight").c_str(), {gdn_conv, conv_k}, &w.gdn_conv_w))
                    return false;
                if (!bind_vec((g + "A_log").c_str(), {n_v_heads}, &w.gdn_a_log)) return false;
                if (!bind_vec((g + "dt_bias").c_str(), {n_v_heads}, &w.gdn_dt_bias)) return false;
                if (!bind_vec((g + "norm.weight").c_str(), {v_head_dim}, &w.gdn_norm))
                    return false;
            } else if (is_qwen35) {
                // ---- Qwen3.5 full attention 层：无 bias，q_proj 携带输出门 ----
                const std::string s = p + "self_attn.";
                if (!bind_mat((s + "q_proj.weight").c_str(), {2 * qd, hidden}, &w.q_proj))
                    return false;
                if (!bind_mat((s + "k_proj.weight").c_str(), {kvd, hidden}, &w.k_proj))
                    return false;
                if (!bind_mat((s + "v_proj.weight").c_str(), {kvd, hidden}, &w.v_proj))
                    return false;
                if (!bind_mat((s + "o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                    return false;
                if (!bind_vec((s + "q_norm.weight").c_str(), {cfg.head_dim}, &w.q_norm))
                    return false;
                if (!bind_vec((s + "k_norm.weight").c_str(), {cfg.head_dim}, &w.k_norm))
                    return false;
            } else {
                // ---- Qwen2.x 层：q/k/v 带 bias ----
                if (!bind_mat((p + "self_attn.q_proj.weight").c_str(), {qd, hidden}, &w.q_proj))
                    return false;
                if (!bind_mat((p + "self_attn.k_proj.weight").c_str(), {kvd, hidden}, &w.k_proj))
                    return false;
                if (!bind_mat((p + "self_attn.v_proj.weight").c_str(), {kvd, hidden}, &w.v_proj))
                    return false;
                if (!bind_vec((p + "self_attn.q_proj.bias").c_str(), {qd}, &w.q_bias))
                    return false;
                if (!bind_vec((p + "self_attn.k_proj.bias").c_str(), {kvd}, &w.k_bias))
                    return false;
                if (!bind_vec((p + "self_attn.v_proj.bias").c_str(), {kvd}, &w.v_bias))
                    return false;
                if (!bind_mat((p + "self_attn.o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                    return false;
            }
        }

        // KV cache 只给 full attention 层（qwen35：n_layers / interval 层）。
        m->kv_.init(cfg.n_full_layers(), static_cast<int>(cfg.n_kv_heads), max_seq_len,
                    static_cast<int>(cfg.head_dim));
        if (is_qwen35) {
            const int n_linear = static_cast<int>(cfg.n_layers) - cfg.n_full_layers();
            m->gdn_state_.init(n_linear, static_cast<int>(cfg.linear_num_v_heads),
                               static_cast<int>(cfg.linear_qk_head_dim),
                               static_cast<int>(cfg.linear_v_head_dim), m->gdn_conv_dim_,
                               static_cast<int>(cfg.linear_conv_kernel_dim));
        }

        // 一次性开好所有 workspace，forward 里不再分配。
        m->hidden_.resize(hidden);
        m->normed_.resize(hidden);
        m->q_.resize(m->q_dim_);
        m->k_.resize(m->kv_dim_);
        m->v_.resize(m->kv_dim_);
        m->attn_.resize(m->q_dim_);
        m->o_.resize(hidden);
        m->gate_.resize(inter);
        m->up_.resize(inter);
        m->ffn_.resize(hidden);
        m->logits_.resize(vocab);
        if (is_qwen35) {
            m->q_full_.resize(2 * static_cast<size_t>(m->q_dim_));
            m->q_gate_.resize(m->q_dim_);
            m->mixed_.resize(m->gdn_conv_dim_);
            m->z_.resize(m->gdn_value_dim_);
            m->b_.resize(cfg.linear_num_v_heads);
            m->a_.resize(cfg.linear_num_v_heads);
            m->gdn_out_.resize(m->gdn_value_dim_);
        }

        *out = std::move(m);
        return true;
    }

    void QwenModel::reset() {
        kv_.reset();
        if (gdn_state_.initialized()) gdn_state_.reset();
        token_count_ = 0;
    }

    // 对单个 token 做一次完整前向。
    // pos 是当前 KV 长度（= 这是序列里第几个位置）；调用结束后 cache 有 pos+1 条，
    // 返回值是 greedy 的下一个 token。数学定义见 docs/qwen_forward.md。
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
                rmsnorm(hidden_.data(), w.input_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
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
                    causal_conv1d_update(mixed, gdn_state_.conv(li), w.gdn_conv_w, mixed,
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
                        l2norm_inplace(qh, qk_hd, 1e-6f);
                        for (int d = 0; d < qk_hd; ++d) qh[d] *= q_scale;
                        l2norm_inplace(kh, qk_hd, 1e-6f);
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
                        gdn_step(S + static_cast<size_t>(h) * qk_hd * v_hd, qh, kh, vh, g,
                                 beta, gdn_out_.data() + h * v_hd, qk_hd, v_hd);
                    }
                }
                // 2f. 门控 RMSNorm：y = rmsnorm(o) * w * silu(z)，按 v 头维度。
                {
                    ScopedTimer t(prof, scope("layer_%d.gdn_norm", i));
                    for (int h = 0; h < n_v_heads; ++h) {
                        rmsnorm_gated(gdn_out_.data() + h * v_hd, z_.data() + h * v_hd,
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
                        rmsnorm(q_.data() + h * head_dim, w.q_norm, q_.data() + h * head_dim,
                                head_dim, cfg_.rms_norm_eps);
                    }
                    for (int h = 0; h < n_kv_heads; ++h) {
                        rmsnorm(k_.data() + h * head_dim, w.k_norm, k_.data() + h * head_dim,
                                head_dim, cfg_.rms_norm_eps);
                    }
                }
                // 2d. partial RoPE：只旋转每头前 rotary_dim 维（rotate-half）。
                {
                    ScopedTimer t(prof, scope("layer_%d.rope", i));
                    partial_rope_ref(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim,
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
                    attention_decode(q_.data(), kv_.k(ci), kv_.v(ci), pos + 1, max_seq_len_,
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
                    rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim, pos,
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
                    attention_decode(q_.data(), kv_.k(static_cast<int>(i)),
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
                rmsnorm(hidden_.data(), w.post_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
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
                swiglu(gate_.data(), up_.data(), inter);
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
            rmsnorm(hidden_.data(), final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
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
    // ---- Batched prefill: GEMM for linear projections, per-token attention ----

    void QwenModel::mm(const void *w, const float *x, float *y, int M, int K, int N) const {
        if (dtype_ == Dtype::kI4) {
            matmul_i4(static_cast<const uint8_t *>(w), x, y, M, K, N, group_size_);
        } else if (dtype_ == Dtype::kF16) {
            // f16 matmul not yet implemented — fall back to N matvecs
            for (int c = 0; c < N; ++c)
                matvec_f16(static_cast<const uint16_t *>(w), x + static_cast<size_t>(c) * K,
                           y + static_cast<size_t>(c) * M, M, K);
        } else {
            matmul_f32(static_cast<const float *>(w), x, y, M, K, N);
        }
    }

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
                    rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.input_ln,
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
                    rope(qc, kc, n_heads, n_kv_heads, head_dim, pos, cfg_.rope_theta);

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
                    attention_decode(qc, kv_.k(static_cast<int>(li)),
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
                    rmsnorm(hid_batch.data() + static_cast<size_t>(c) * hidden, w.post_ln,
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
                    swiglu(gate_batch.data() + static_cast<size_t>(c) * inter,
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
            rmsnorm(last_hidden, final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
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
