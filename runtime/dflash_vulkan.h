#pragma once

#include <memory>
#include <string>
#include <vector>

namespace tinyqwen {

  class DFlashModel;

  // GPU-resident speculative engine. DFlash proposal and dense Qwen3 target
  // verification are separate one-submit passes on one device, sharing the
  // target lm_head and allocator-backed resident buffers.
  class DFlashVulkanEngine {
  public:
    static std::unique_ptr<DFlashVulkanEngine> create(const DFlashModel &model, std::string *err);
    ~DFlashVulkanEngine();

    bool propose(const float *target_hidden, int ctx_tokens, int confirmed_seq, int anchor,
                 int block_tokens, std::vector<int> *proposals, std::string *err);

    // CPU prefill remains the numerical anchor.  This imports its confirmed KV
    // prefix once; all later target verification runs in this engine/device.
    bool initialize_target_from_cpu(std::string *err);
    bool verify_target(const int *tokens, int n, std::vector<int> *all_next,
                       std::vector<float> *captured_hidden, std::string *err);
    bool truncate_target(int seq_len, std::string *err);
    void reset_target();
    int target_seq_len() const;
    const std::string &device_name() const;

  private:
    struct Impl;
    explicit DFlashVulkanEngine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

} // namespace tinyqwen
