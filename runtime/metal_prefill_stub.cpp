// ============================================================================
// 文件: metal_prefill_stub.cpp
// 作用: 非 Apple 平台的 metal_prefill_* 空实现（链接占位）
//
// 为什么需要?
//   metal_prefill.mm 只在 APPLE 构建时编译（见 runtime/CMakeLists.txt）。
//   main.cpp 无条件引用这几个符号，所以其他平台（Linux / Android NDK）
//   必须有一份定义，否则链接失败。用 stub 而不是在 main.cpp 里写
//   #ifdef __APPLE__ —— 保持编排层代码干净、分支收敛在这一个文件里。
//
// 行为:
//   available() 返回 false，main.cpp 据此拒绝 --engine metal 并给出明确报错，
//   而不是崩溃或静默走错路径。
// ============================================================================

#include <string>

#include "metal_prefill.h"

namespace tinyqwen {
    bool metal_prefill_available() { return false; }

    bool metal_prefill_create(const ModelFile *, int, std::string *err, MetalPrefillEngine **) {
        if (err) *err = "Metal prefill 仅在 Apple 平台可用";
        return false;
    }

    void metal_prefill_destroy(MetalPrefillEngine *) {}

    int metal_prefill_run(MetalPrefillEngine *, const int *, int, float *, KvCache *,
                          std::string *err) {
        if (err) *err = "Metal prefill 仅在 Apple 平台可用";
        return -1;
    }

    std::string metal_prefill_device_name(const MetalPrefillEngine *) { return std::string(); }
} // namespace tinyqwen
