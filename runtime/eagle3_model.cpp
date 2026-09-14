#include "eagle3_model.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>

#include "backend_cpu.h"
#include "ref_ops.h"

namespace tinyqwen {
namespace {

bool shape_is(const TensorView *tensor, Dtype dtype,
              std::initializer_list<uint64_t> dims) {
    if (!tensor || tensor->dtype != dtype ||
        tensor->ndim != static_cast<int>(dims.size())) {
        return false;
    }
    int i = 0;
    for (uint64_t dim : dims) {
        if (tensor->shape[i++] != dim) return false;
    }
    return true;
}

const TensorView *need(const ModelFile &file, const std::string &name, Dtype dtype,
                       std::initializer_list<uint64_t> dims, std::string *err) {
    const TensorView *tensor = file.get(name);
    if (!shape_is(tensor, dtype, dims)) {
        if (err) *err = "EAGLE3 tensor missing, wrong dtype, or wrong shape: " + name;
        return nullptr;
    }
    return tensor;
}

WeightTensor f16_weight(const TensorView *tensor, int rows, int cols) {
    return WeightTensor{tensor ? tensor->data : nullptr, QuantType::kF16,
                        rows, cols, 0};
}

} // namespace

bool Eagle3Model::create(const ModelFile &file, int max_seq_len, QwenModel &target,
                         std::string *err, std::unique_ptr<Eagle3Model> *out,
                         std::unique_ptr<IBackend> backend) {
    if (!out) return false;
    out->reset();
    const ModelConfig &cfg = file.config();
    const ModelConfig &target_cfg = target.config();
    if (file.header().version < 2 ||
        file.header().dtype != static_cast<uint32_t>(Dtype::kF16)) {
        if (err) *err = "EAGLE3 file must be a v2+ FP16 tinyqwen file";
        return false;
    }
    if (cfg.n_layers != 1 || cfg.hidden_size == 0 || cfg.intermediate_size == 0 ||
        cfg.n_heads == 0 || cfg.n_kv_heads == 0 || cfg.head_dim == 0 ||
        cfg.n_heads % cfg.n_kv_heads != 0 ||
        cfg.n_heads * cfg.head_dim != 2 * cfg.hidden_size ||
        cfg.n_kv_heads * cfg.head_dim != cfg.hidden_size) {
        if (err) *err = "unsupported EAGLE3 layer configuration";
        return false;
    }
    if (cfg.hidden_size != target_cfg.hidden_size ||
        cfg.vocab_size != target_cfg.vocab_size) {
        if (err) *err = "EAGLE3 and target hidden/vocabulary sizes differ";
        return false;
    }
    if (max_seq_len <= 0 || max_seq_len > static_cast<int>(cfg.max_seq_len)) {
        if (err) *err = "invalid EAGLE3 max_seq_len";
        return false;
    }

    const int h = static_cast<int>(cfg.hidden_size);
    const int intermediate = static_cast<int>(cfg.intermediate_size);
    const int q_dim = static_cast<int>(cfg.n_heads * cfg.head_dim);
    const int kv_dim = static_cast<int>(cfg.n_kv_heads * cfg.head_dim);

    const TensorView *fc = need(file, "fc.weight", Dtype::kF16,
                                {cfg.hidden_size, 3 * cfg.hidden_size}, err);
    const TensorView *hidden_norm = need(file, "midlayer.hidden_norm.weight", Dtype::kF16,
                                         {cfg.hidden_size}, err);
    const TensorView *input_norm = need(file, "midlayer.input_layernorm.weight", Dtype::kF16,
                                        {cfg.hidden_size}, err);
    const TensorView *q = need(file, "midlayer.self_attn.q_proj.weight", Dtype::kF16,
                               {static_cast<uint64_t>(q_dim), 2 * cfg.hidden_size}, err);
    const TensorView *k = need(file, "midlayer.self_attn.k_proj.weight", Dtype::kF16,
                               {static_cast<uint64_t>(kv_dim), 2 * cfg.hidden_size}, err);
    const TensorView *v = need(file, "midlayer.self_attn.v_proj.weight", Dtype::kF16,
                               {static_cast<uint64_t>(kv_dim), 2 * cfg.hidden_size}, err);
    const TensorView *o = need(file, "midlayer.self_attn.o_proj.weight", Dtype::kF16,
                               {cfg.hidden_size, static_cast<uint64_t>(q_dim)}, err);
    const TensorView *post_norm = need(file, "midlayer.post_attention_layernorm.weight",
                                       Dtype::kF16, {cfg.hidden_size}, err);
    const TensorView *gate = need(file, "midlayer.mlp.gate_proj.weight", Dtype::kF16,
                                  {cfg.intermediate_size, cfg.hidden_size}, err);
    const TensorView *up = need(file, "midlayer.mlp.up_proj.weight", Dtype::kF16,
                                {cfg.intermediate_size, cfg.hidden_size}, err);
    const TensorView *down = need(file, "midlayer.mlp.down_proj.weight", Dtype::kF16,
                                  {cfg.hidden_size, cfg.intermediate_size}, err);
    const TensorView *final_norm = need(file, "norm.weight", Dtype::kF16,
                                        {cfg.hidden_size}, err);
    const TensorView *lm_head = file.get("lm_head.weight");
    if (!fc || !hidden_norm || !input_norm || !q || !k || !v || !o || !post_norm ||
        !gate || !up || !down || !final_norm) {
        return false;
    }
    if (!lm_head || lm_head->dtype != Dtype::kF16 || lm_head->ndim != 2 ||
        lm_head->shape[1] != cfg.hidden_size || lm_head->shape[0] == 0 ||
        lm_head->shape[0] > cfg.vocab_size) {
        if (err) *err = "missing or invalid EAGLE3 lm_head.weight";
        return false;
    }
    const int draft_vocab = static_cast<int>(lm_head->shape[0]);

    const TensorView *mapping = need(file, "eagle3.draft_to_target", Dtype::kF16,
                                     {static_cast<uint64_t>(draft_vocab), 2}, err);
    const TensorView *layer_ids = file.get("eagle3.target_layer_ids");
    if (!mapping) return false;
    if (!layer_ids || layer_ids->dtype != Dtype::kF16 || layer_ids->ndim != 1 ||
        layer_ids->shape[0] != 3) {
        if (err) *err = "missing or invalid eagle3.target_layer_ids";
        return false;
    }

    auto model = std::unique_ptr<Eagle3Model>(new Eagle3Model());
    model->target_ = &target;
    model->backend_ = backend ? std::move(backend) : create_cpu_backend();
    model->cfg_ = cfg;
    model->max_seq_len_ = max_seq_len;
    model->draft_vocab_size_ = draft_vocab;
    model->fc_ = f16_weight(fc, h, 3 * h);
    model->q_proj_ = f16_weight(q, q_dim, 2 * h);
    model->k_proj_ = f16_weight(k, kv_dim, 2 * h);
    model->v_proj_ = f16_weight(v, kv_dim, 2 * h);
    model->o_proj_ = f16_weight(o, h, q_dim);
    model->gate_proj_ = f16_weight(gate, intermediate, h);
    model->up_proj_ = f16_weight(up, intermediate, h);
    model->down_proj_ = f16_weight(down, h, intermediate);
    model->lm_head_ = f16_weight(lm_head, draft_vocab, h);

    model->norm_storage_.reserve(4);
    const auto bind_norm = [&](const TensorView *tensor) -> const float * {
        model->norm_storage_.emplace_back(static_cast<size_t>(h));
        std::vector<float> &dst = model->norm_storage_.back();
        const auto *src = reinterpret_cast<const uint16_t *>(tensor->data);
        for (int i = 0; i < h; ++i) dst[i] = half_to_float(src[i]);
        return dst.data();
    };
    model->hidden_norm_weight_ = bind_norm(hidden_norm);
    model->input_norm_weight_ = bind_norm(input_norm);
    model->post_norm_weight_ = bind_norm(post_norm);
    model->final_norm_weight_ = bind_norm(final_norm);

    const auto *layer_values = reinterpret_cast<const uint16_t *>(layer_ids->data);
    for (int i = 0; i < 3; ++i) {
        const float value = half_to_float(layer_values[i]);
        const int id = static_cast<int>(std::nearbyint(value));
        if (!std::isfinite(value) || value != static_cast<float>(id) || id < 0 ||
            id >= static_cast<int>(target_cfg.n_layers) ||
            (!model->target_layer_ids_.empty() && id <= model->target_layer_ids_.back())) {
            if (err) *err = "invalid EAGLE3 target layer id metadata";
            return false;
        }
        model->target_layer_ids_.push_back(id);
    }

    const auto *map_values = reinterpret_cast<const uint16_t *>(mapping->data);
    model->draft_to_target_.reserve(draft_vocab);
    std::vector<uint8_t> seen(cfg.vocab_size, 0);
    for (int i = 0; i < draft_vocab; ++i) {
        const float high_value = half_to_float(map_values[static_cast<size_t>(i) * 2]);
        const float low_value = half_to_float(map_values[static_cast<size_t>(i) * 2 + 1]);
        const int high = static_cast<int>(std::nearbyint(high_value));
        const int low = static_cast<int>(std::nearbyint(low_value));
        const int id = high * 256 + low;
        if (!std::isfinite(high_value) || !std::isfinite(low_value) ||
            high_value != static_cast<float>(high) ||
            low_value != static_cast<float>(low) || high < 0 || low < 0 || low >= 256 ||
            id < 0 ||
            id >= static_cast<int>(cfg.vocab_size) || seen[id]) {
            if (err) *err = "invalid EAGLE3 draft-to-target vocabulary map";
            return false;
        }
        seen[id] = 1;
        model->draft_to_target_.push_back(id);
    }

    const size_t cache_elements = static_cast<size_t>(max_seq_len) * kv_dim;
    model->k_cache_.resize(cache_elements);
    model->v_cache_.resize(cache_elements);
#if !defined(__aarch64__) && !defined(_M_ARM64)
    model->k_cache_f32_.resize(cache_elements);
    model->v_cache_f32_.resize(cache_elements);
#endif
    model->residual_.resize(h);
    model->hidden_normed_.resize(h);
    model->embedding_.resize(h);
    model->embedding_normed_.resize(h);
    model->attention_input_.resize(2 * h);
    model->q_.resize(q_dim);
    model->k_.resize(kv_dim);
    model->v_.resize(kv_dim);
    model->attention_.resize(q_dim);
    model->projected_.resize(h);
    model->post_normed_.resize(h);
    model->gate_.resize(intermediate);
    model->up_.resize(intermediate);
    model->ffn_.resize(h);
    model->last_hidden_.resize(h);
    model->final_normed_.resize(h);
    model->logits_.resize(draft_vocab);
    *out = std::move(model);
    return true;
}

void Eagle3Model::reset() {
    seq_len_ = 0;
    next_proposal_ = -1;
    proposal_ready_ = false;
}

void Eagle3Model::norm(const float *x, const float *weight, float *out, int n) const {
    backend_->rmsnorm(x, weight, out, n, cfg_.rms_norm_eps);
}

bool Eagle3Model::run_layer(const float *hidden_in, int token_id, std::string *err) {
    if (!hidden_in || token_id < 0 || token_id >= static_cast<int>(cfg_.vocab_size) ||
        seq_len_ >= max_seq_len_) {
        if (err) *err = "invalid EAGLE3 recurrent step or exhausted draft KV capacity";
        return false;
    }
    const int h = static_cast<int>(cfg_.hidden_size);
    const int intermediate = static_cast<int>(cfg_.intermediate_size);
    const int q_dim = static_cast<int>(cfg_.n_heads * cfg_.head_dim);
    const int kv_dim = static_cast<int>(cfg_.n_kv_heads * cfg_.head_dim);
    const int n_heads = static_cast<int>(cfg_.n_heads);
    const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
    const int head_dim = static_cast<int>(cfg_.head_dim);

    std::copy(hidden_in, hidden_in + h, residual_.begin());
    norm(residual_.data(), hidden_norm_weight_, hidden_normed_.data(), h);
    if (!target_->copy_embedding(token_id, embedding_.data())) {
        if (err) *err = "failed to read EAGLE3 token embedding from the target";
        return false;
    }
    norm(embedding_.data(), input_norm_weight_, embedding_normed_.data(), h);
    std::copy(embedding_normed_.begin(), embedding_normed_.end(), attention_input_.begin());
    std::copy(hidden_normed_.begin(), hidden_normed_.end(), attention_input_.begin() + h);

    backend_->matvec_qkv(q_proj_, k_proj_, v_proj_, attention_input_.data(),
                         q_.data(), k_.data(), v_.data(), q_dim, kv_dim, 2 * h);
    backend_->rope(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim,
                   seq_len_, cfg_.rope_theta);
    for (int head = 0; head < n_kv_heads; ++head) {
        const size_t cache_offset =
            (static_cast<size_t>(head) * max_seq_len_ + seq_len_) * head_dim;
        const size_t value_offset = static_cast<size_t>(head) * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            k_cache_[cache_offset + i] = float_to_half(k_[value_offset + i]);
            v_cache_[cache_offset + i] = float_to_half(v_[value_offset + i]);
#if !defined(__aarch64__) && !defined(_M_ARM64)
            k_cache_f32_[cache_offset + i] = k_[value_offset + i];
            v_cache_f32_[cache_offset + i] = v_[value_offset + i];
#endif
        }
    }

#if defined(__aarch64__) || defined(_M_ARM64)
    backend_->attention_decode_f16kv(
        q_.data(), k_cache_.data(), v_cache_.data(), seq_len_ + 1, max_seq_len_,
        n_heads, n_kv_heads, head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim)), attention_.data());
#else
    // The optimized FP16-KV kernel is ARM-only. Other hosts use the persistent
    // FP32 mirror so the token loop remains allocation-free.
    backend_->attention_decode(
        q_.data(), k_cache_f32_.data(), v_cache_f32_.data(), seq_len_ + 1,
        max_seq_len_,
        n_heads, n_kv_heads, head_dim,
        1.0f / std::sqrt(static_cast<float>(head_dim)), attention_.data());
#endif

    backend_->matvec(o_proj_, attention_.data(), projected_.data(), h, q_dim);
    for (int i = 0; i < h; ++i) projected_[i] += residual_[i];
    norm(projected_.data(), post_norm_weight_, post_normed_.data(), h);
    backend_->matvec_pair(gate_proj_, up_proj_, post_normed_.data(),
                          gate_.data(), up_.data(), intermediate, h);
    backend_->swiglu(gate_.data(), up_.data(), intermediate);
    backend_->matvec(down_proj_, gate_.data(), ffn_.data(), h, intermediate);
    for (int i = 0; i < h; ++i) last_hidden_[i] = projected_[i] + ffn_[i];
    ++seq_len_;
    return true;
}

int Eagle3Model::project_next() const {
    const int h = static_cast<int>(cfg_.hidden_size);
    norm(last_hidden_.data(), final_norm_weight_, final_normed_.data(), h);
    backend_->matvec(lm_head_, final_normed_.data(), logits_.data(),
                     draft_vocab_size_, h);
    const int draft_id = backend_->argmax(logits_.data(), draft_vocab_size_);
    if (draft_id < 0 || draft_id >= static_cast<int>(draft_to_target_.size())) return -1;
    return draft_to_target_[draft_id];
}

bool Eagle3Model::extend_target(const float *target_hidden, int n,
                                const int *next_tokens, int *next_proposal,
                                std::string *err) {
    if (!target_hidden || !next_tokens || n <= 0 || seq_len_ + n > max_seq_len_) {
        if (err) *err = "invalid EAGLE3 target extension dimensions";
        return false;
    }
    const int h = static_cast<int>(cfg_.hidden_size);
    for (int row = 0; row < n; ++row) {
        backend_->matvec(fc_, target_hidden + static_cast<size_t>(row) * 3 * h,
                         projected_.data(), h, 3 * h);
        if (!run_layer(projected_.data(), next_tokens[row], err)) return false;
    }
    next_proposal_ = project_next();
    if (next_proposal_ < 0) {
        if (err) *err = "EAGLE3 lm_head argmax failed";
        return false;
    }
    proposal_ready_ = true;
    if (next_proposal) *next_proposal = next_proposal_;
    return true;
}

bool Eagle3Model::propose(int count, std::vector<int> *proposals, std::string *err) {
    if (!proposals || count <= 0 || !proposal_ready_ ||
        seq_len_ + count - 1 > max_seq_len_) {
        if (err) *err = "EAGLE3 proposals require a prepared state and available KV capacity";
        return false;
    }
    proposals->clear();
    proposals->reserve(count);
    proposals->push_back(next_proposal_);
    for (int i = 1; i < count; ++i) {
        if (!run_layer(last_hidden_.data(), proposals->back(), err)) return false;
        next_proposal_ = project_next();
        if (next_proposal_ < 0) {
            if (err) *err = "EAGLE3 recurrent lm_head argmax failed";
            return false;
        }
        proposals->push_back(next_proposal_);
    }
    return true;
}

bool Eagle3Model::truncate(int seq_len, std::string *err) {
    if (seq_len < 0 || seq_len > seq_len_) {
        if (err) *err = "invalid EAGLE3 KV truncation length";
        return false;
    }
    seq_len_ = seq_len;
    next_proposal_ = -1;
    proposal_ready_ = false;
    return true;
}

} // namespace tinyqwen
