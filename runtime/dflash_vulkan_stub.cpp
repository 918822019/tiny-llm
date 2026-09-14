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

  const std::string &DFlashVulkanEngine::device_name() const {
    static const std::string empty;
    return empty;
  }

} // namespace tinyqwen
