#include "dflash_vulkan.h"

#include <utility>

namespace tinyqwen {

  std::unique_ptr<DFlashVulkanEngine> DFlashVulkanEngine::create(const DFlashModel &,
                                                                 std::string *err) {
    if (err)
      *err = "DFlash Vulkan engine is available only in an Android Vulkan build";
    return nullptr;
  }

  struct DFlashVulkanEngine::Impl {};

  DFlashVulkanEngine::DFlashVulkanEngine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
  DFlashVulkanEngine::~DFlashVulkanEngine() = default;

  bool DFlashVulkanEngine::propose(const float *, int, int, int, int, std::vector<int> *,
                                   std::string *err) {
    if (err)
      *err = "DFlash Vulkan engine is unavailable";
    return false;
  }

  bool DFlashVulkanEngine::initialize_target_from_cpu(std::string *err) {
    if (err)
      *err = "DFlash Vulkan engine is unavailable";
    return false;
  }

  bool DFlashVulkanEngine::verify_target(const int *, int, std::vector<int> *,
                                         std::vector<float> *, std::string *err) {
    if (err)
      *err = "DFlash Vulkan engine is unavailable";
    return false;
  }

  bool DFlashVulkanEngine::truncate_target(int, std::string *err) {
    if (err)
      *err = "DFlash Vulkan engine is unavailable";
    return false;
  }

  void DFlashVulkanEngine::reset_target() {}

  int DFlashVulkanEngine::target_seq_len() const { return -1; }

  const std::string &DFlashVulkanEngine::device_name() const {
    static const std::string empty;
    return empty;
  }

} // namespace tinyqwen
