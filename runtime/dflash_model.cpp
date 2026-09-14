#include "dflash_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "backend_cpu.h"
#include "dflash_vulkan.h"
#include "ref_ops.h"

namespace tinyqwen {
namespace {
bool shape_is(const TensorView *t, std::initializer_list<uint64_t> dims) {
    if (!t || t->dtype != Dtype::kF16 || t->ndim != static_cast<int>(dims.size())) return false;
    int i = 0;
    for (uint64_t d : dims) if (t->shape[i++] != d) return false;
    return true;
}

const uint16_t *need(const ModelFile &f, const std::string &name,
                     std::initializer_list<uint64_t> dims, std::string *err) {
    const TensorView *t = f.get(name);
    if (!shape_is(t, dims)) {
        if (err) *err = "DFlash tensor missing or wrong shape: " + name;
        return nullptr;
    }
    return reinterpret_cast<const uint16_t *>(t->data);
}
} // namespace

DFlashModel::~DFlashModel() = default;

bool DFlashModel::create(const ModelFile &file, int max_seq_len, QwenModel &target,
                         std::string *err, std::unique_ptr<DFlashModel> *out,
                         std::unique_ptr<IBackend> backend) {
    if (!out) return false;
    out->reset();
    const ModelConfig &c = file.config();
    if (file.header().version < 2 || file.header().dtype != static_cast<uint32_t>(Dtype::kF16)) {
        if (err) *err = "DFlash file must be a v2+ FP16 tiny file";
        return false;
    }
    if (c.hidden_size != target.config().hidden_size ||
        c.vocab_size != target.config().vocab_size || c.n_layers == 0 ||
        c.n_heads * c.head_dim == 0 || c.n_kv_heads * c.head_dim != c.hidden_size) {
        if (err) *err = "DFlash/target config mismatch";
        return false;
    }

    auto m = std::unique_ptr<DFlashModel>(new DFlashModel());
    m->target_ = &target;
    m->backend_ = backend ? std::move(backend) : create_cpu_backend();
    m->cfg_ = c;
    m->max_seq_len_ = max_seq_len;
    m->mask_token_id_ = static_cast<int>(c.eos_token_id);
    const int H = static_cast<int>(c.hidden_size);
    const int I = static_cast<int>(c.intermediate_size);
    const int Q = static_cast<int>(c.n_heads * c.head_dim);
    const int KV = static_cast<int>(c.n_kv_heads * c.head_dim);
    const int HD = static_cast<int>(c.head_dim);

    const TensorView *ids = file.get("dflash.target_layer_ids");
    if (!ids || ids->dtype != Dtype::kF16 || ids->ndim != 1 || ids->shape[0] == 0) {
        if (err) *err = "missing dflash.target_layer_ids";
        return false;
    }
    const auto *idp = reinterpret_cast<const uint16_t *>(ids->data);
    for (uint64_t i = 0; i < ids->shape[0]; ++i)
        m->target_layer_ids_.push_back(static_cast<int>(half_to_float(idp[i])));
    const int K = static_cast<int>(m->target_layer_ids_.size());

    const TensorView *bp = file.get("block_pos_embed.weight");
    if (!bp || bp->dtype != Dtype::kF16 || bp->ndim != 2 || bp->shape[1] != c.hidden_size) {
        if (err) *err = "missing block_pos_embed.weight";
        return false;
    }
    m->block_size_ = static_cast<int>(bp->shape[0]);
    m->block_pos_ = reinterpret_cast<const uint16_t *>(bp->data);
    m->fusion_logits_ = need(file, "fusion_logits", {c.n_layers, static_cast<uint64_t>(K)}, err);
    const uint16_t *final_norm_f16 = need(file, "norm.weight", {c.hidden_size}, err);
    const TensorView *mw1 = file.get("markov_w1.weight");
    const TensorView *mw2 = file.get("markov_w2.weight");
    if (!mw1 || !mw2 || mw1->dtype != Dtype::kF16 || mw2->dtype != Dtype::kF16 ||
        mw1->ndim != 2 || mw2->ndim != 2 || mw1->shape[0] != c.vocab_size ||
        mw2->shape[0] != c.vocab_size || mw1->shape[1] != mw2->shape[1]) {
        if (err) *err = "missing or invalid Markov head";
        return false;
    }
    m->markov_rank_ = static_cast<int>(mw1->shape[1]);
    m->markov_w1_ = reinterpret_cast<const uint16_t *>(mw1->data);
    m->markov_w2_ = reinterpret_cast<const uint16_t *>(mw2->data);
    if (!m->fusion_logits_ || !final_norm_f16) return false;

    m->norm_storage_.reserve(static_cast<size_t>(c.n_layers) * 4 + 1);
    const auto bind_norm = [&](const uint16_t *src, int n) -> const float * {
        if (!src) return nullptr;
        m->norm_storage_.emplace_back(static_cast<size_t>(n));
        auto &dst = m->norm_storage_.back();
        for (int i = 0; i < n; ++i) dst[i] = half_to_float(src[i]);
        return dst.data();
    };
    m->final_norm_ = bind_norm(final_norm_f16, H);

    m->layers_.resize(c.n_layers);
    for (uint32_t li = 0; li < c.n_layers; ++li) {
        const std::string p = "layers." + std::to_string(li) + ".";
        Layer &w = m->layers_[li];
        w.input_norm = bind_norm(need(file, p + "input_layernorm.weight", {c.hidden_size}, err), H);
        w.q_proj = need(file, p + "self_attn.q_proj.weight", {static_cast<uint64_t>(Q), c.hidden_size}, err);
        w.k_proj = need(file, p + "self_attn.k_proj.weight", {static_cast<uint64_t>(KV), c.hidden_size}, err);
        w.v_proj = need(file, p + "self_attn.v_proj.weight", {static_cast<uint64_t>(KV), c.hidden_size}, err);
        w.k_proj_ctx = need(file, p + "self_attn.k_proj_ctx.weight", {static_cast<uint64_t>(KV), c.hidden_size}, err);
        w.v_proj_ctx = need(file, p + "self_attn.v_proj_ctx.weight", {static_cast<uint64_t>(KV), c.hidden_size}, err);
        w.q_norm = bind_norm(need(file, p + "self_attn.q_norm.weight", {static_cast<uint64_t>(HD)}, err), HD);
        w.k_norm = bind_norm(need(file, p + "self_attn.k_norm.weight", {static_cast<uint64_t>(HD)}, err), HD);
        w.o_proj = need(file, p + "self_attn.o_proj.weight", {c.hidden_size, static_cast<uint64_t>(Q)}, err);
        w.post_norm = bind_norm(need(file, p + "post_attention_layernorm.weight", {c.hidden_size}, err), H);
        w.gate = need(file, p + "mlp.gate_proj.weight", {static_cast<uint64_t>(I), c.hidden_size}, err);
        w.up = need(file, p + "mlp.up_proj.weight", {static_cast<uint64_t>(I), c.hidden_size}, err);
        w.down = need(file, p + "mlp.down_proj.weight", {c.hidden_size, static_cast<uint64_t>(I)}, err);
        if (!w.input_norm || !w.q_proj || !w.k_proj || !w.v_proj || !w.k_proj_ctx ||
            !w.v_proj_ctx || !w.q_norm || !w.k_norm || !w.o_proj || !w.post_norm ||
            !w.gate || !w.up || !w.down) return false;
    }
    const size_t cache_elems = static_cast<size_t>(c.n_layers) * max_seq_len * KV;
    m->k_cache_.resize(cache_elems);
    m->v_cache_.resize(cache_elems);
    *out = std::move(m);
    return true;
}

void DFlashModel::reset() {
    seq_len_ = 0;
    if (vulkan_) vulkan_->reset_target();
}

bool DFlashModel::enable_vulkan(std::string *err) {
    vulkan_ = DFlashVulkanEngine::create(*this, err);
    if (!vulkan_) return false;
    std::fprintf(stderr, "[vulkan] DFlash + target GPU-resident decode: %s\n",
                 vulkan_->device_name().c_str());
    return true;
}

bool DFlashModel::initialize_vulkan_target(std::string *err) {
    if (!vulkan_) {
        if (err) *err = "DFlash Vulkan engine is not enabled";
        return false;
    }
    return vulkan_->initialize_target_from_cpu(err);
}

bool DFlashModel::verify_vulkan_target(const int *tokens, int n,
                                       std::vector<int> *all_next,
                                       std::vector<float> *captured_hidden,
                                       std::string *err) {
    if (!vulkan_) {
        if (err) *err = "DFlash Vulkan engine is not enabled";
        return false;
    }
    return vulkan_->verify_target(tokens, n, all_next, captured_hidden, err);
}

bool DFlashModel::truncate_vulkan_target(int seq_len, std::string *err) {
    if (!vulkan_) {
        if (err) *err = "DFlash Vulkan engine is not enabled";
        return false;
    }
    return vulkan_->truncate_target(seq_len, err);
}

int DFlashModel::vulkan_target_seq_len() const {
    return vulkan_ ? vulkan_->target_seq_len() : -1;
}

void DFlashModel::mv(const uint16_t *w, const float *x, float *y,
                     int rows, int cols) const {
    backend_->matvec(WeightTensor{w, QuantType::kF16, rows, cols, 0},
                     x, y, rows, cols);
}

void DFlashModel::mm(const uint16_t *w, const float *x, float *y,
                     int rows, int cols, int tokens) const {
    backend_->matmul(WeightTensor{w, QuantType::kF16, rows, cols, 0},
                     x, y, rows, cols, tokens);
}

void DFlashModel::norm(const float *x, const float *w, float *y, int n) const {
    backend_->rmsnorm(x, w, y, n, cfg_.rms_norm_eps);
}

void DFlashModel::rope(float *q, float *k, int pos) const {
    backend_->rope(q, k, static_cast<int>(cfg_.n_heads),
                   static_cast<int>(cfg_.n_kv_heads),
                   static_cast<int>(cfg_.head_dim), pos, cfg_.rope_theta);
}

void DFlashModel::attention(const float *q, const std::vector<float> &noise_k,
                            const std::vector<float> &noise_v, int layer,
                            int total_ctx, int noise_tokens, float *out) const {
    const int NH = static_cast<int>(cfg_.n_heads);
    const int NK = static_cast<int>(cfg_.n_kv_heads);
    const int HD = static_cast<int>(cfg_.head_dim);
    const int KV = NK * HD;
    const int groups = NH / NK;
    const float scale = 1.0f / std::sqrt(static_cast<float>(HD));
    const size_t layer_off = static_cast<size_t>(layer) * max_seq_len_ * KV;
    std::vector<float> accum(static_cast<size_t>(NH) * HD, 0.0f);
    for (int h = 0; h < NH; ++h) {
        const int kh = h / groups;
        const float *qh = q + h * HD;
        float m = -INFINITY, denom = 0.0f;
        float *acc = accum.data() + static_cast<size_t>(h) * HD;
        const int total = total_ctx + noise_tokens;
        for (int t = 0; t < total; ++t) {
            float score = 0.0f;
            if (t < total_ctx) {
                const size_t off = layer_off + static_cast<size_t>(t) * KV + kh * HD;
                for (int d = 0; d < HD; ++d)
                    score += qh[d] * half_to_float(k_cache_[off + d]);
            } else {
                const size_t off = static_cast<size_t>(t - total_ctx) * KV + kh * HD;
                const float *kf = noise_k.data() + off;
                for (int d = 0; d < HD; ++d) score += qh[d] * kf[d];
            }
            score *= scale;
            const float nm = std::max(m, score);
            const float old_scale = std::exp(m - nm);
            const float new_scale = std::exp(score - nm);
            denom = denom * old_scale + new_scale;
            if (t < total_ctx) {
                const size_t off = layer_off + static_cast<size_t>(t) * KV + kh * HD;
                for (int d = 0; d < HD; ++d)
                    acc[d] = acc[d] * old_scale +
                             half_to_float(v_cache_[off + d]) * new_scale;
            } else {
                const size_t off = static_cast<size_t>(t - total_ctx) * KV + kh * HD;
                const float *vf = noise_v.data() + off;
                for (int d = 0; d < HD; ++d)
                    acc[d] = acc[d] * old_scale + vf[d] * new_scale;
            }
            m = nm;
        }
        for (int d = 0; d < HD; ++d) out[h * HD + d] = acc[d] / denom;
    }
}

bool DFlashModel::propose(const float *target_hidden, int ctx_tokens, int anchor,
                          int block_tokens, std::vector<int> *proposals,
                          std::string *err) {
    if (!target_hidden || !proposals || ctx_tokens <= 0 || block_tokens < 2 ||
        block_tokens > block_size_ || seq_len_ + ctx_tokens + block_tokens > max_seq_len_) {
        if (err) *err = "invalid DFlash proposal dimensions";
        return false;
    }
    if (vulkan_) {
        const bool ok = vulkan_->propose(target_hidden, ctx_tokens, seq_len_, anchor,
                                         block_tokens, proposals, err);
        if (ok) seq_len_ += ctx_tokens;
        return ok;
    }
    proposals->clear();
    const int H = static_cast<int>(cfg_.hidden_size);
    const int I = static_cast<int>(cfg_.intermediate_size);
    const int Q = static_cast<int>(cfg_.n_heads * cfg_.head_dim);
    const int KV = static_cast<int>(cfg_.n_kv_heads * cfg_.head_dim);
    const int NH = static_cast<int>(cfg_.n_heads);
    const int NK = static_cast<int>(cfg_.n_kv_heads);
    const int HD = static_cast<int>(cfg_.head_dim);
    const int V = static_cast<int>(cfg_.vocab_size);
    const int K = static_cast<int>(target_layer_ids_.size());
    const int start = seq_len_ + ctx_tokens;

    std::vector<float> hidden(static_cast<size_t>(block_tokens) * H);
    for (int t = 0; t < block_tokens; ++t) {
        if (!target_->copy_embedding(t == 0 ? anchor : mask_token_id_,
                                     hidden.data() + static_cast<size_t>(t) * H)) {
            if (err) *err = "failed to read target token embedding";
            return false;
        }
        for (int d = 0; d < H; ++d)
            hidden[static_cast<size_t>(t) * H + d] +=
                half_to_float(block_pos_[static_cast<size_t>(t) * H + d]);
    }

    std::vector<float> fused(static_cast<size_t>(ctx_tokens) * H);
    std::vector<float> normed(static_cast<size_t>(block_tokens) * H);
    std::vector<float> q(static_cast<size_t>(block_tokens) * Q);
    std::vector<float> nk(static_cast<size_t>(block_tokens) * KV);
    std::vector<float> nv(static_cast<size_t>(block_tokens) * KV);
    std::vector<float> attn(static_cast<size_t>(block_tokens) * Q);
    std::vector<float> o_batch(static_cast<size_t>(block_tokens) * H);
    std::vector<float> gate(static_cast<size_t>(block_tokens) * I);
    std::vector<float> up(static_cast<size_t>(block_tokens) * I);
    std::vector<float> ffn(static_cast<size_t>(block_tokens) * H);
    std::vector<float> kctx(static_cast<size_t>(ctx_tokens) * KV);
    std::vector<float> vctx(static_cast<size_t>(ctx_tokens) * KV);
    std::vector<float> qdummy(Q, 0.0f);

    for (uint32_t li = 0; li < cfg_.n_layers; ++li) {
        const Layer &w = layers_[li];
        std::vector<float> alpha(K);
        float mx = -INFINITY, sum = 0.0f;
        for (int k = 0; k < K; ++k) mx = std::max(mx, half_to_float(fusion_logits_[li * K + k]));
        for (int k = 0; k < K; ++k) { alpha[k] = std::exp(half_to_float(fusion_logits_[li * K + k]) - mx); sum += alpha[k]; }
        for (float &a : alpha) a /= sum;
        for (int c = 0; c < ctx_tokens; ++c) {
            float *dst = fused.data() + static_cast<size_t>(c) * H;
            std::fill(dst, dst + H, 0.0f);
            for (int k = 0; k < K; ++k) {
                const float *src = target_hidden + (static_cast<size_t>(c) * K + k) * H;
                for (int d = 0; d < H; ++d) dst[d] += alpha[k] * src[d];
            }
        }
        mm(w.k_proj_ctx, fused.data(), kctx.data(), KV, H, ctx_tokens);
        mm(w.v_proj_ctx, fused.data(), vctx.data(), KV, H, ctx_tokens);
        for (int c = 0; c < ctx_tokens; ++c) {
            float *kc = kctx.data() + static_cast<size_t>(c) * KV;
            float *vc = vctx.data() + static_cast<size_t>(c) * KV;
            for (int h = 0; h < NK; ++h)
                norm(kc + h * HD, w.k_norm, kc + h * HD, HD);
            std::fill(qdummy.begin(), qdummy.end(), 0.0f);
            rope(qdummy.data(), kc, seq_len_ + c);
            const size_t off = (static_cast<size_t>(li) * max_seq_len_ + seq_len_ + c) * KV;
            for (int d = 0; d < KV; ++d) {
                k_cache_[off + d] = float_to_half(kc[d]);
                v_cache_[off + d] = float_to_half(vc[d]);
            }
        }

        for (int t = 0; t < block_tokens; ++t) {
            float *z = normed.data() + static_cast<size_t>(t) * H;
            norm(hidden.data() + static_cast<size_t>(t) * H, w.input_norm, z, H);
        }
        mm(w.q_proj, normed.data(), q.data(), Q, H, block_tokens);
        mm(w.k_proj, normed.data(), nk.data(), KV, H, block_tokens);
        mm(w.v_proj, normed.data(), nv.data(), KV, H, block_tokens);
        for (int t = 0; t < block_tokens; ++t) {
            float *qt = q.data() + static_cast<size_t>(t) * Q;
            float *kt = nk.data() + static_cast<size_t>(t) * KV;
            for (int h = 0; h < NH; ++h) norm(qt + h * HD, w.q_norm, qt + h * HD, HD);
            for (int h = 0; h < NK; ++h) norm(kt + h * HD, w.k_norm, kt + h * HD, HD);
            rope(qt, kt, start + t);
        }
        for (int t = 0; t < block_tokens; ++t) {
            attention(q.data() + static_cast<size_t>(t) * Q, nk, nv,
                      static_cast<int>(li), start, block_tokens,
                      attn.data() + static_cast<size_t>(t) * Q);
        }
        mm(w.o_proj, attn.data(), o_batch.data(), H, Q, block_tokens);
        for (int t = 0; t < block_tokens; ++t) {
            const size_t ho = static_cast<size_t>(t) * H;
            for (int d = 0; d < H; ++d) hidden[ho + d] += o_batch[ho + d];
            norm(hidden.data() + ho, w.post_norm, normed.data() + ho, H);
        }
        mm(w.gate, normed.data(), gate.data(), I, H, block_tokens);
        mm(w.up, normed.data(), up.data(), I, H, block_tokens);
        for (int t = 0; t < block_tokens; ++t)
            backend_->swiglu(gate.data() + static_cast<size_t>(t) * I,
                             up.data() + static_cast<size_t>(t) * I, I);
        mm(w.down, gate.data(), ffn.data(), H, I, block_tokens);
        for (size_t d = 0; d < static_cast<size_t>(block_tokens) * H; ++d)
            hidden[d] += ffn[d];
    }
    seq_len_ = start;

    const int draft_positions = block_tokens - 1;
    std::vector<float> final_h(static_cast<size_t>(draft_positions) * H);
    std::vector<float> logits(static_cast<size_t>(draft_positions) * V);
    std::vector<float> markov(markov_rank_), bias(V);
    for (int t = 1; t < block_tokens; ++t)
        norm(hidden.data() + static_cast<size_t>(t) * H, final_norm_,
             final_h.data() + static_cast<size_t>(t - 1) * H, H);
    target_->project_lm_head_raw_batch(final_h.data(), logits.data(), draft_positions);
    int prev = anchor;
    for (int t = 1; t < block_tokens; ++t) {
        const uint16_t *row = markov_w1_ + static_cast<size_t>(prev) * markov_rank_;
        for (int r = 0; r < markov_rank_; ++r) markov[r] = half_to_float(row[r]);
        mv(markov_w2_, markov.data(), bias.data(), V, markov_rank_);
        float *position_logits = logits.data() + static_cast<size_t>(t - 1) * V;
        for (int v = 0; v < V; ++v) position_logits[v] += bias[v];
        prev = backend_->argmax(position_logits, V);
        proposals->push_back(prev);
    }
    return true;
}

} // namespace tinyqwen
