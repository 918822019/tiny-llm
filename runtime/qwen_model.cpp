#include "qwen_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

#include "ref_ops.h"

namespace tinyqwen {

namespace {

void top_k_logits(const float* logits, int vocab, int k, TopKResult* out) {
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

std::string shape_str(const std::vector<uint64_t>& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) r += ", ";
    r += std::to_string(s[i]);
  }
  return r + "]";
}

}  // namespace

const float* QwenModel::require(const ModelFile& file, const std::string& name,
                                const std::vector<uint64_t>& shape, std::string* err) {
  const TensorView* t = file.get(name);
  if (!t) {
    if (err) *err = "missing tensor: " + name;
    return nullptr;
  }
  if (t->dtype != Dtype::kF32) {
    if (err) *err = "tensor " + name + " is not f32";
    return nullptr;
  }
  if (static_cast<size_t>(t->ndim) != shape.size()) {
    if (err) *err = "tensor " + name + " ndim mismatch: got " + std::to_string(t->ndim) +
                    " expected " + std::to_string(shape.size());
    return nullptr;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (t->shape[i] != shape[i]) {
      if (err) *err = "tensor " + name + " shape mismatch: got " + shape_str(
                          std::vector<uint64_t>(t->shape, t->shape + t->ndim)) +
                      " expected " + shape_str(shape);
      return nullptr;
    }
  }
  return t->f32();
}

bool QwenModel::create(const ModelFile& file, int max_seq_len, Profiler& profiler,
                       std::string* err, std::unique_ptr<QwenModel>* out) {
  out->reset();
  if (!file.loaded()) {
    if (err) *err = "model file not loaded";
    return false;
  }
  const ModelConfig& cfg = file.config();

  if (max_seq_len <= 0) max_seq_len = static_cast<int>(cfg.max_seq_len);
  if (max_seq_len > static_cast<int>(cfg.max_seq_len)) {
    if (err) *err = "max_seq_len " + std::to_string(max_seq_len) +
                    " exceeds model max " + std::to_string(cfg.max_seq_len);
    return false;
  }

  std::unique_ptr<QwenModel> m(new QwenModel());
  m->cfg_ = cfg;
  m->profiler_ = &profiler;
  m->max_seq_len_ = max_seq_len;
  m->q_dim_ = static_cast<int>(cfg.n_heads * cfg.head_dim);
  m->kv_dim_ = static_cast<int>(cfg.n_kv_heads * cfg.head_dim);

  const uint64_t hidden = cfg.hidden_size;
  const uint64_t inter = cfg.intermediate_size;
  const uint64_t vocab = cfg.vocab_size;
  const uint64_t qd = static_cast<uint64_t>(m->q_dim_);
  const uint64_t kvd = static_cast<uint64_t>(m->kv_dim_);

  m->embed_ = m->require(file, "model.embed_tokens.weight", {vocab, hidden}, err);
  if (!m->embed_) return false;
  m->final_norm_ = m->require(file, "model.norm.weight", {hidden}, err);
  if (!m->final_norm_) return false;
  if (cfg.tied_embeddings) {
    m->lm_head_ = m->embed_;
  } else {
    m->lm_head_ = m->require(file, "lm_head.weight", {vocab, hidden}, err);
    if (!m->lm_head_) return false;
  }

  m->layers_.resize(cfg.n_layers);
  for (uint32_t i = 0; i < cfg.n_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    LayerWeights& w = m->layers_[i];
    w.input_ln = m->require(file, p + "input_layernorm.weight", {hidden}, err);
    if (!w.input_ln) return false;
    w.q_proj = m->require(file, p + "self_attn.q_proj.weight", {qd, hidden}, err);
    if (!w.q_proj) return false;
    w.k_proj = m->require(file, p + "self_attn.k_proj.weight", {kvd, hidden}, err);
    if (!w.k_proj) return false;
    w.v_proj = m->require(file, p + "self_attn.v_proj.weight", {kvd, hidden}, err);
    if (!w.v_proj) return false;
    w.q_bias = m->require(file, p + "self_attn.q_proj.bias", {qd}, err);
    if (!w.q_bias) return false;
    w.k_bias = m->require(file, p + "self_attn.k_proj.bias", {kvd}, err);
    if (!w.k_bias) return false;
    w.v_bias = m->require(file, p + "self_attn.v_proj.bias", {kvd}, err);
    if (!w.v_bias) return false;
    w.o_proj = m->require(file, p + "self_attn.o_proj.weight", {hidden, qd}, err);
    if (!w.o_proj) return false;
    w.post_ln = m->require(file, p + "post_attention_layernorm.weight", {hidden}, err);
    if (!w.post_ln) return false;
    w.gate = m->require(file, p + "mlp.gate_proj.weight", {inter, hidden}, err);
    if (!w.gate) return false;
    w.up = m->require(file, p + "mlp.up_proj.weight", {inter, hidden}, err);
    if (!w.up) return false;
    w.down = m->require(file, p + "mlp.down_proj.weight", {hidden, inter}, err);
    if (!w.down) return false;
  }

  m->kv_.init(static_cast<int>(cfg.n_layers), static_cast<int>(cfg.n_kv_heads), max_seq_len,
              static_cast<int>(cfg.head_dim));

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

int QwenModel::forward_token(int token_id, TopKResult* topk, int topk_k) {
  const int hidden = static_cast<int>(cfg_.hidden_size);
  const int inter = static_cast<int>(cfg_.intermediate_size);
  const int vocab = static_cast<int>(cfg_.vocab_size);
  const int n_heads = static_cast<int>(cfg_.n_heads);
  const int n_kv_heads = static_cast<int>(cfg_.n_kv_heads);
  const int head_dim = static_cast<int>(cfg_.head_dim);
  const int pos = kv_.seq_len();

  if (token_id < 0 || token_id >= vocab) {
    std::fprintf(stderr, "tinyqwen: token_id %d out of range [0, %d)\n", token_id, vocab);
    std::abort();
  }
  if (pos >= max_seq_len_) {
    std::fprintf(stderr, "tinyqwen: position %d exceeds max_seq_len %d\n", pos, max_seq_len_);
    std::abort();
  }

  Profiler& prof = *profiler_;
  prof.begin_token(token_count_, pos, token_count_ < prompt_len_);

  char name[64];
  const auto scope = [&](const char* fmt, int layer) {
    std::snprintf(name, sizeof(name), fmt, layer);
    return name;
  };

  // ---- embedding ----
  {
    ScopedTimer t(prof, "embed");
    std::memcpy(hidden_.data(), embed_ + static_cast<size_t>(token_id) * hidden,
                hidden * sizeof(float));
  }

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (uint32_t i = 0; i < cfg_.n_layers; ++i) {
    const LayerWeights& w = layers_[i];

    {
      ScopedTimer t(prof, scope("layer_%d.input_layernorm", i));
      rmsnorm_ref(hidden_.data(), w.input_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.q_proj", i));
      matvec_f32_ref(w.q_proj, normed_.data(), q_.data(), q_dim_, hidden);
      for (int j = 0; j < q_dim_; ++j) q_[j] += w.q_bias[j];
    }
    {
      ScopedTimer t(prof, scope("layer_%d.k_proj", i));
      matvec_f32_ref(w.k_proj, normed_.data(), k_.data(), kv_dim_, hidden);
      for (int j = 0; j < kv_dim_; ++j) k_[j] += w.k_bias[j];
    }
    {
      ScopedTimer t(prof, scope("layer_%d.v_proj", i));
      matvec_f32_ref(w.v_proj, normed_.data(), v_.data(), kv_dim_, hidden);
      for (int j = 0; j < kv_dim_; ++j) v_[j] += w.v_bias[j];
    }
    {
      ScopedTimer t(prof, scope("layer_%d.rope", i));
      rope_ref(q_.data(), k_.data(), n_heads, n_kv_heads, head_dim, pos, cfg_.rope_theta);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.kv_append", i));
      const size_t pos_off = static_cast<size_t>(pos) * head_dim;
      const size_t head_plane = static_cast<size_t>(max_seq_len_) * head_dim;
      float* k_layer = kv_.k(static_cast<int>(i));
      float* v_layer = kv_.v(static_cast<int>(i));
      for (int h = 0; h < n_kv_heads; ++h) {
        std::memcpy(k_layer + h * head_plane + pos_off, k_.data() + h * head_dim,
                    head_dim * sizeof(float));
        std::memcpy(v_layer + h * head_plane + pos_off, v_.data() + h * head_dim,
                    head_dim * sizeof(float));
      }
    }
    {
      ScopedTimer t(prof, scope("layer_%d.attention", i));
      attention_decode_ref(q_.data(), kv_.k(static_cast<int>(i)),
                           kv_.v(static_cast<int>(i)), pos + 1, max_seq_len_, n_heads,
                           n_kv_heads, head_dim, attn_scale, attn_.data());
    }
    {
      ScopedTimer t(prof, scope("layer_%d.o_proj", i));
      matvec_f32_ref(w.o_proj, attn_.data(), o_.data(), hidden, q_dim_);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.residual_attn", i));
      for (int j = 0; j < hidden; ++j) hidden_[j] += o_[j];
    }
    {
      ScopedTimer t(prof, scope("layer_%d.post_attn_layernorm", i));
      rmsnorm_ref(hidden_.data(), w.post_ln, normed_.data(), hidden, cfg_.rms_norm_eps);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.gate_proj", i));
      matvec_f32_ref(w.gate, normed_.data(), gate_.data(), inter, hidden);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.up_proj", i));
      matvec_f32_ref(w.up, normed_.data(), up_.data(), inter, hidden);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.swiglu", i));
      silu_ref(gate_.data(), gate_.data(), inter);
      for (int j = 0; j < inter; ++j) gate_[j] *= up_[j];
    }
    {
      ScopedTimer t(prof, scope("layer_%d.down_proj", i));
      matvec_f32_ref(w.down, gate_.data(), ffn_.data(), hidden, inter);
    }
    {
      ScopedTimer t(prof, scope("layer_%d.residual_ffn", i));
      for (int j = 0; j < hidden; ++j) hidden_[j] += ffn_[j];
    }
  }

  {
    ScopedTimer t(prof, "final_norm");
    rmsnorm_ref(hidden_.data(), final_norm_, normed_.data(), hidden, cfg_.rms_norm_eps);
  }
  {
    ScopedTimer t(prof, "lm_head");
    matvec_f32_ref(lm_head_, normed_.data(), logits_.data(), vocab, hidden);
  }

  int next = 0;
  {
    ScopedTimer t(prof, "topk_argmax");
    if (topk) {
      top_k_logits(logits_.data(), vocab, topk_k, topk);
      next = topk->indices.empty() ? 0 : topk->indices[0];
    } else {
      next = argmax_ref(logits_.data(), vocab);
    }
  }

  kv_.advance(1);
  token_count_ += 1;
  prof.end_token();
  return next;
}

}  // namespace tinyqwen
