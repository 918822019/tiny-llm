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

#include "backend_cpu.h"  // create_cpu_backend
#include "dispatch.h"     // matvec_f32 通用入口（分发到 _ref / 将来的优化版）
#include "gdn_ops.h"      // Qwen3.5 GDN 算子（l2norm / conv1d / delta rule / ...）
#include "ref_ops.h"

namespace tinyqwen {
    namespace {
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

    // matvec 分派薄封装：通过 backend 调用具体实现
    void QwenModel::mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const {
        WeightTensor wt{w, static_cast<QuantType>(dtype_), out_dim, in_dim, group_size_};
        backend_->matvec(wt, x, y, out_dim, in_dim);
    }

    void QwenModel::mv_pair(const void *w1, const void *w2, const float *x, float *y1,
                            float *y2, int out_dim, int in_dim) const {
        WeightTensor wt1{w1, static_cast<QuantType>(dtype_), out_dim, in_dim, group_size_};
        WeightTensor wt2{w2, static_cast<QuantType>(dtype_), out_dim, in_dim, group_size_};
        backend_->matvec_pair(wt1, wt2, x, y1, y2, out_dim, in_dim);
    }

    void QwenModel::mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                           float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                           int in_dim) const {
        WeightTensor wqt{wq, static_cast<QuantType>(dtype_), q_dim, in_dim, group_size_};
        WeightTensor wkt{wk, static_cast<QuantType>(dtype_), kv_dim, in_dim, group_size_};
        WeightTensor wvt{wv, static_cast<QuantType>(dtype_), kv_dim, in_dim, group_size_};
        backend_->matvec_qkv(wqt, wkt, wvt, x, yq, yk, yv, q_dim, kv_dim, in_dim);
    }

    // 工厂：绑定权重视图、校验每个 tensor、初始化 KV cache 和 workspace。
    // 所有失败经 *err 报告；成功后 *out 持有一个可直接运行的模型。
    bool QwenModel::create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out,
                           std::unique_ptr<IBackend> backend) {
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
        // 创建后端：默认 CPU，未来可扩展 CUDA/Metal/...
        m->backend_ = backend ? std::move(backend) : create_cpu_backend();
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
            // tied：lm_head 与词嵌入同源。默认共享 embed（省内存）；但若文件里
            // 带了单独量化/独立的 lm_head.weight（i4 导出选项），优先用它——
            // fp32 embed 每 token 544MB 流量曾是 INT4 路径最大单项开销。
            if (file.get("lm_head.weight") &&
                bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) {
                // 走 dtype 对应的正常 matvec 路径（i4 文件里即 i4 kernel）
            } else {
                m->lm_head_ = m->embed_;
                // I4 文件中 embed 存为 fp32（lookup table 不量化），lm_head 投影需走 f32 路径。
                if (m->dtype_ == Dtype::kI4) m->lm_head_is_f32_ = true;
            }
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
    // ---- Batched prefill: GEMM for linear projections, per-token attention ----

    void QwenModel::mm(const void *w, const float *x, float *y, int M, int K, int N) const {
        WeightTensor wt{w, static_cast<QuantType>(dtype_), M, K, group_size_};
        backend_->matmul(wt, x, y, M, K, N);
    }

} // namespace tinyqwen

