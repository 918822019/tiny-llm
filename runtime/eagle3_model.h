#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "model_loader.h"
#include "qwen_model.h"

namespace tinyqwen {

// One-layer EAGLE3 greedy drafter used by the SpecForge Qwen3 checkpoints.
// The target owns token embeddings and supplies three captured residual streams;
// this class owns only the trained projection/layer/head and its FP16 KV cache.
class Eagle3Model {
public:
    static bool create(const ModelFile &file, int max_seq_len, QwenModel &target,
                       std::string *err, std::unique_ptr<Eagle3Model> *out,
                       std::unique_ptr<IBackend> backend = nullptr);

    void reset();
    int seq_len() const { return seq_len_; }
    int max_seq_len() const { return max_seq_len_; }
    int draft_vocab_size() const { return draft_vocab_size_; }
    const std::vector<int> &target_layer_ids() const { return target_layer_ids_; }

    // Consume confirmed target residual streams. target_hidden is token-major
    // [n, 3, hidden], and next_tokens contains the shifted token embedding paired
    // with each row. The final row prepares the first proposal for the next tree.
    bool extend_target(const float *target_hidden, int n, const int *next_tokens,
                       int *next_proposal, std::string *err);

    // Return count chained proposals. The first was prepared by extend_target;
    // each later proposal consumes the preceding proposal through the recurrent
    // EAGLE3 layer and temporarily extends the draft KV cache.
    bool propose(int count, std::vector<int> *proposals, std::string *err);

    // Discard a speculative KV suffix. A subsequent extend_target call is
    // required before proposals can be generated again.
    bool truncate(int seq_len, std::string *err = nullptr);

private:
    Eagle3Model() = default;

    bool run_layer(const float *hidden_in, int token_id, std::string *err);
    int project_next() const;
    void norm(const float *x, const float *weight, float *out, int n) const;

    QwenModel *target_ = nullptr;
    std::unique_ptr<IBackend> backend_;
    ModelConfig cfg_{};
    int max_seq_len_ = 0;
    int seq_len_ = 0;
    int draft_vocab_size_ = 0;
    int next_proposal_ = -1;
    bool proposal_ready_ = false;
    std::vector<int> target_layer_ids_;
    std::vector<int> draft_to_target_;

    WeightTensor fc_{};
    WeightTensor q_proj_{};
    WeightTensor k_proj_{};
    WeightTensor v_proj_{};
    WeightTensor o_proj_{};
    WeightTensor gate_proj_{};
    WeightTensor up_proj_{};
    WeightTensor down_proj_{};
    WeightTensor lm_head_{};
    const float *hidden_norm_weight_ = nullptr;
    const float *input_norm_weight_ = nullptr;
    const float *post_norm_weight_ = nullptr;
    const float *final_norm_weight_ = nullptr;
    std::vector<std::vector<float>> norm_storage_;

    // Draft KV is head-major [n_kv_heads, max_seq_len, head_dim], matching
    // the runtime attention kernels' cache contract.
    std::vector<uint16_t> k_cache_;
    std::vector<uint16_t> v_cache_;
    // Non-ARM portability fallback. ARM uses the compact FP16 cache above
    // directly; other hosts keep a persistent FP32 mirror for reference attention.
    std::vector<float> k_cache_f32_;
    std::vector<float> v_cache_f32_;

    // Persistent fp32 activation workspace; no allocation occurs in a token step.
    std::vector<float> residual_;
    std::vector<float> hidden_normed_;
    std::vector<float> embedding_;
    std::vector<float> embedding_normed_;
    std::vector<float> attention_input_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> attention_;
    std::vector<float> projected_;
    std::vector<float> post_normed_;
    std::vector<float> gate_;
    std::vector<float> up_;
    std::vector<float> ffn_;
    std::vector<float> last_hidden_;
    mutable std::vector<float> final_normed_;
    mutable std::vector<float> logits_;
};

} // namespace tinyqwen
