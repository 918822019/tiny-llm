#pragma once

#include <memory>
#include <string>

#include "backend.h"

namespace tinyqwen {

  // Android Vulkan FP16 weight-only backend.  The implementation accelerates
  // matvec/matmul and inherits the already verified CPU implementations for the
  // remaining operators.  Non-Android builds provide a stub so callers do not
  // need platform preprocessor branches.
  bool vulkan_backend_available();
  std::unique_ptr<IBackend> create_vulkan_backend(std::string *err = nullptr);

} // namespace tinyqwen
