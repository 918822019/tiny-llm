#pragma once

#include <memory>
#include <string>
#include <vector>

namespace tinyqwen {

  class DFlashModel;

  // Whole-proposal Vulkan engine.  All three DFlash transformer layers, shared
  // target lm_head, Markov correction and argmax are submitted as one command
  // buffer, leaving only one CPU/GPU synchronization per speculative block.
  class DFlashVulkanEngine {
  public:
    static std::unique_ptr<DFlashVulkanEngine> create(const DFlashModel &model, std::string *err);
    ~DFlashVulkanEngine();

    bool propose(const float *target_hidden, int ctx_tokens, int confirmed_seq, int anchor,
                 int block_tokens, std::vector<int> *proposals, std::string *err);
    const std::string &device_name() const;

  private:
    struct Impl;
    explicit DFlashVulkanEngine(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
  };

} // namespace tinyqwen
