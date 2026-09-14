#include "backend_vulkan.h"

namespace tinyqwen {

  bool vulkan_backend_available() {
    return false;
  }

  std::unique_ptr<IBackend> create_vulkan_backend(std::string *err) {
    if (err)
      *err = "Vulkan backend is available only in an Android Vulkan build";
    return nullptr;
  }

} // namespace tinyqwen
