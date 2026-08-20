// 量化抽象层的实现

#include "quant.h"

namespace tinyqwen {
namespace quantization {

    bool parse_type(const std::string& str, Type* out) {
        if (str == "f32" || str == "fp32") {
            *out = Type::kF32;
            return true;
        }
        if (str == "f16" || str == "fp16") {
            *out = Type::kF16;
            return true;
        }
        if (str == "i4" || str == "int4") {
            *out = Type::kI4;
            return true;
        }
        return false;
    }

    const char* type_name(Type type) {
        switch (type) {
            case Type::kF32: return "f32";
            case Type::kF16: return "f16";
            case Type::kI4:  return "i4";
            default:         return "unknown";
        }
    }

    bool is_supported(Type type) {
        switch (type) {
            case Type::kF32:
            case Type::kF16:
            case Type::kI4:
                return true;
            default:
                return false;
        }
    }

    size_t quantized_size(Type type, int rows, int cols, int group_size) {
        switch (type) {
            case Type::kF32:
                return static_cast<size_t>(rows) * cols * 4;
            case Type::kF16:
                return static_cast<size_t>(rows) * cols * 2;
            case Type::kI4: {
                // INT4: 每组 1 byte（2 个 4-bit 值） + scale (fp16) + zero (fp16)
                const int num_groups = (cols + group_size - 1) / group_size;
                const size_t data_bytes = static_cast<size_t>(rows) * ((cols + 1) / 2);
                const size_t param_bytes = static_cast<size_t>(rows) * num_groups * 2 * 2;
                return data_bytes + param_bytes;
            }
            default:
                return 0;
        }
    }

} // namespace quantization
} // namespace tinyqwen
