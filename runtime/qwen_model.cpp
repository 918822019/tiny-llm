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
        if (t->dtype != dtype_) {
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
        if (dtype_ == Dtype::kF32) return t->f32();
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
        if (dtype_ == Dtype::kF16) {
            matvec_f16(static_cast<const uint16_t *>(w), x, y, out_dim, in_dim);
        } else {
            matvec_f32(static_cast<const float *>(w), x, y, out_dim, in_dim);
        }
    }

    void QwenModel::mv_pair(const void *w1, const void *w2, const float *x, float *y1,
                            float *y2, int out_dim, int in_dim) const {
        if (dtype_ == Dtype::kF16) {
            matvec_pair_f16(static_cast<const uint16_t *>(w1),
                            static_cast<const uint16_t *>(w2), x, y1, y2, out_dim, in_dim);
        } else {
            matvec_pair_f32(static_cast<const float *>(w1), static_cast<const float *>(w2), x,
                            y1, y2, out_dim, in_dim);
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
        m->dtype_ = static_cast<Dtype>(file.header().dtype); // f32 / f16，决定 matvec 入口
        m->profiler_ = &profiler;
        m->max_seq_len_ = max_seq_len;
        m->q_dim_ = static_cast<int>(cfg.n_heads * cfg.head_dim);
        m->kv_dim_ = static_cast<int>(cfg.n_kv_heads * cfg.head_dim);

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
        } else {
            if (!bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) return false;
        }

        // 每层权重。tensor 名字完全沿用 HuggingFace 约定，缺哪个名字
        // 就能直接定位到 exporter 的问题。
        m->layers_.resize(cfg.n_layers);
        for (uint32_t i = 0; i < cfg.n_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            LayerWeights &w = m->layers_[i];
            if (!bind_vec((p + "input_layernorm.weight").c_str(), {hidden}, &w.input_ln))
                return false;
            if (!bind_mat((p + "self_attn.q_proj.weight").c_str(), {qd, hidden}, &w.q_proj))
                return false;
            if (!bind_mat((p + "self_attn.k_proj.weight").c_str(), {kvd, hidden}, &w.k_proj))
                return false;
            if (!bind_mat((p + "self_attn.v_proj.weight").c_str(), {kvd, hidden}, &w.v_proj))
                return false;
            if (!bind_vec((p + "self_attn.q_proj.bias").c_str(), {qd}, &w.q_bias)) return false;
            if (!bind_vec((p + "self_attn.k_proj.bias").c_str(), {kvd}, &w.k_bias)) return false;
            if (!bind_vec((p + "self_attn.v_proj.bias").c_str(), {kvd}, &w.v_bias)) return false;
            if (!bind_mat((p + "self_attn.o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                return false;
            if (!bind_vec((p + "post_attention_layernorm.weight").c_str(), {hidden}, &w.post_ln))
                return false;
            if (!bind_mat((p + "mlp.gate_proj.weight").c_str(), {inter, hidden}, &w.gate))
                return false;
            if (!bind_mat((p + "mlp.up_proj.weight").c_str(), {inter, hidden}, &w.up))
                return false;
            if (!bind_mat((p + "mlp.down_proj.weight").c_str(), {hidden, inter}, &w.down))
                return false;
        }

        m->kv_.init(static_cast<int>(cfg.n_layers), static_cast<int>(cfg.n_kv_heads), max_seq_len,
                    static_cast<int>(cfg.head_dim));

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

        *out = std::move(m);
        return true;
    }

    void QwenModel::reset() {
        kv_.reset();
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
            if (dtype_ == Dtype::kF32) {
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
        for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
            const LayerWeights &w = layers_[i];

            // ---- attention 块：norm -> qkv(+bias) -> RoPE -> 入 cache -> attend ----

            // 2a. attention 前的 RMSNorm。
            {
                ScopedTimer t(prof, scope("layer_%d.input_layernorm", i));
                rmsnorm_ref(hidden_.data(), w.input_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }
            // 2b. q/k/v 投影 + bias。Qwen2/2.5 的 q/k/v 有 bias，且必须加在 RoPE 之前。
            // matvec = 矩阵乘向量：normed_ 过投影矩阵得到 q_/k_/v_。
            {
                ScopedTimer t(prof, scope("layer_%d.q_proj", i));
                mv(w.q_proj, normed_.data(), q_.data(), q_dim_, hidden);
                for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
            }
            {
                // k/v 投影合并成一次成对 matvec：两个小矩阵共享同一个输入
                // normed_。dispatch 兜底语义 = 分开调两次（未注册 pair 的
                // impl 行为不变）；注册了 pair 的 impl（neon_mt 系）可以把
                // 两次 0.45MB 的内联小调用合成一次更大的调用去摊薄/并行。
                ScopedTimer t(prof, scope("layer_%d.kv_proj", i));
                mv_pair(w.k_proj, w.v_proj, normed_.data(), k_.data(), v_.data(),
                        kv_dim_, hidden);
                for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
                for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
            }
            // 2c. RoPE 旋转位置编码：把"位置 pos"的信息编进 q/k（v 不需要）。
            {
                ScopedTimer t(prof, scope("layer_%d.rope", i));
                rope_ref(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim, pos, cfg_.rope_theta);
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
                attention_decode_ref(q_.data(), kv_.k(static_cast<int>(i)),
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

            // ---- FFN 块（SwiGLU）：norm -> gate/up -> silu*up -> down ----

            // 2h. FFN 前的 RMSNorm。
            {
                ScopedTimer t(prof, scope("layer_%d.post_attn_layernorm", i));
                rmsnorm_ref(hidden_.data(), w.post_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
            }
            // 2i. gate 和 up 两个投影并行（SwiGLU 需要两条支路）。
            {
                ScopedTimer t(prof, scope("layer_%d.gate_proj", i));
                mv(w.gate, normed_.data(), gate_.data(), inter, hidden);
            }
            {
                ScopedTimer t(prof, scope("layer_%d.up_proj", i));
                mv(w.up, normed_.data(), up_.data(), inter, hidden);
            }
            // 2j. SwiGLU 融合：SiLU 只作用在 gate 支路，再和 up 逐元素相乘。
            // gate_ 就地复用为融合结果，直接喂给 down_proj。
            {
                ScopedTimer t(prof, scope("layer_%d.swiglu", i));
                silu_ref(gate_.data(), gate_.data(), inter);
                for (int j = 0; j < inter; ++j) gate_[j] *= up_[j];
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
            rmsnorm_ref(hidden_.data(), final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
        }
        {
            // lm_head：把 hidden 向量投成 vocab 维的 logits（每个词一个分数）。
            // tied 时 lm_head_ 就是 embed_（见 create）。
            ScopedTimer t(prof, "lm_head");
            mv(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
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
                next = argmax_ref(logits_.data(), vocab);
            }
        }

        // 提交本 token 写入的 cache 槽位，收尾 profiler 记录。
        kv_.advance(1);
        token_count_ += 1;
        prof.end_token();
        return next;
    }
} // namespace tinyqwen
