#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "model_loader.h"
#include "profiler.h"

namespace tinyqwen {

struct TopKResult {
  std::vector<int> indices;
  std::vector<float> values;
};

// Fixed-structure Qwen-like decoder-only model.
// batch = 1, token-by-token, greedy, fp32 reference path.
//
// No graph abstraction: one hardcoded forward with explicit buffers, so the
// computation order matches docs/qwen_forward.md 1:1 and stays auditable.
class QwenModel {
 public:
  // Validates every required tensor (name + shape) against the header config.
  // max_seq_len: runtime KV capacity; must be <= header max_seq_len.
  static bool create(const ModelFile& file, int max_seq_len, Profiler& profiler,
                     std::string* err, std::unique_ptr<QwenModel>* out);

  // Runs one token at the current position (== kv_cache().seq_len()).
  // Returns the greedy next token id; optionally fills top-k logits (topk_k).
  int forward_token(int token_id, TopKResult* topk = nullptr, int topk_k = 5);

  void reset();  // clears KV cache and token counter
  void set_prompt_len(int n) { prompt_len_ = n; }  // for profiler is_prefill

  const ModelConfig& config() const { return cfg_; }
  KvCache& kv_cache() { return kv_; }
  int token_count() const { return token_count_; }
  // Logits of the most recent forward_token (vocab_size floats).
  const float* last_logits() const { return logits_.data(); }

 private:
  QwenModel() = default;

  struct LayerWeights {
    const float* input_ln = nullptr;
    const float* q_proj = nullptr;
    const float* k_proj = nullptr;
    const float* v_proj = nullptr;
    // Qwen2/2.5 attention has q/k/v biases (attention_bias=True); no o bias.
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    const float* o_proj = nullptr;
    const float* post_ln = nullptr;
    const float* gate = nullptr;
    const float* up = nullptr;
    const float* down = nullptr;
  };

  const float* require(const ModelFile& file, const std::string& name,
                       const std::vector<uint64_t>& shape, std::string* err);

  ModelConfig cfg_{};
  Profiler* profiler_ = nullptr;
  int prompt_len_ = 0;
  int token_count_ = 0;
  int q_dim_ = 0;   // n_heads * head_dim
  int kv_dim_ = 0;  // n_kv_heads * head_dim
  int max_seq_len_ = 0;

  const float* embed_ = nullptr;
  const float* final_norm_ = nullptr;
  const float* lm_head_ = nullptr;  // == embed_ when tied
  std::vector<LayerWeights> layers_;
  KvCache kv_;

  // Workspace buffers, allocated once in create().
  std::vector<float> hidden_, normed_, q_, k_, v_, attn_, o_, gate_, up_, ffn_, logits_;
};

}  // namespace tinyqwen
