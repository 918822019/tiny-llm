#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "model_loader.h"
#include "qwen_model.h"

namespace tinyqwen {

class DFlashVulkanEngine;

class DFlashModel {
public:
    static bool create(const ModelFile &file, int max_seq_len, QwenModel &target,
                       std::string *err, std::unique_ptr<DFlashModel> *out,
                       std::unique_ptr<IBackend> backend = nullptr);

    ~DFlashModel();

    void reset();
    int seq_len() const { return seq_len_; }
    int block_size() const { return block_size_; }
    int mask_token_id() const { return mask_token_id_; }
    const std::vector<int> &target_layer_ids() const { return target_layer_ids_; }
    bool enable_vulkan(std::string *err);
    bool using_vulkan() const { return vulkan_ != nullptr; }

    // With the dedicated Vulkan engine enabled, target decode/verification
    // shares its device, allocator and lm_head with the DFlash proposal pass.
    bool initialize_vulkan_target(std::string *err);
    bool verify_vulkan_target(const int *tokens, int n, std::vector<int> *all_next,
                              std::vector<float> *captured_hidden, std::string *err);
    bool truncate_vulkan_target(int seq_len, std::string *err);
    int vulkan_target_seq_len() const;

    // target_hidden is token-major [ctx_tokens, K, hidden]. The method first
    // commits those confirmed target features to the draft KV cache, then
    // proposes block_tokens-1 tokens after anchor. Noise K/V are never cached.
    bool propose(const float *target_hidden, int ctx_tokens, int anchor,
                 int block_tokens, std::vector<int> *proposals,
                 std::string *err);

private:
    friend class DFlashVulkanEngine;
    struct Layer {
        const float *input_norm = nullptr;
        const uint16_t *q_proj = nullptr;
        const uint16_t *k_proj = nullptr;
        const uint16_t *v_proj = nullptr;
        const uint16_t *k_proj_ctx = nullptr;
        const uint16_t *v_proj_ctx = nullptr;
        const float *q_norm = nullptr;
        const float *k_norm = nullptr;
        const uint16_t *o_proj = nullptr;
        const float *post_norm = nullptr;
        const uint16_t *gate = nullptr;
        const uint16_t *up = nullptr;
        const uint16_t *down = nullptr;
    };

    DFlashModel() = default;
    void mv(const uint16_t *w, const float *x, float *y, int rows, int cols) const;
    void mm(const uint16_t *w, const float *x, float *y,
            int rows, int cols, int tokens) const;
    void norm(const float *x, const float *w, float *y, int n) const;
    void rope(float *q, float *k, int pos) const;
    void attention(const float *q, const std::vector<float> &noise_k,
                   const std::vector<float> &noise_v, int layer, int total_ctx,
                   int noise_tokens, float *out) const;

    QwenModel *target_ = nullptr;
    std::unique_ptr<IBackend> backend_;
    ModelConfig cfg_{};
    int max_seq_len_ = 0;
    int seq_len_ = 0;
    int block_size_ = 0;
    int mask_token_id_ = -1;
    int markov_rank_ = 0;
    std::vector<int> target_layer_ids_;
    std::vector<Layer> layers_;
    const uint16_t *fusion_logits_ = nullptr;
    const uint16_t *block_pos_ = nullptr;
    const float *final_norm_ = nullptr;
    const uint16_t *markov_w1_ = nullptr;
    const uint16_t *markov_w2_ = nullptr;
    std::vector<std::vector<float>> norm_storage_;

    // token-major [layer, max_seq, kv_dim], fp16 storage / fp32 compute.
    std::vector<uint16_t> k_cache_;
    std::vector<uint16_t> v_cache_;
    std::unique_ptr<DFlashVulkanEngine> vulkan_;
};

} // namespace tinyqwen
