// =============================================================================
// quant.cpp
// =============================================================================
// 文件级别说明：
//   本文件是量化抽象层（quantization/quant.h）的实现，提供不同数据类型
//   （FP32、FP16、INT4）的通用接口实现。
//
//   量化抽象层是 tinyqwen 支持多种权重精度的基础：
//   - FP32: 全精度浮点（4 bytes/element），用于正确性基准
//   - FP16: 半精度浮点（2 bytes/element），内存减半，推荐推理精度
//   - INT4: 4-bit 整数（0.5 bytes/element），极致压缩，配合反量化使用
//
// 提供的功能：
//   1. parse_type()      - 将字符串（如 "f16", "int4"）解析为 Type 枚举
//   2. type_name()       - 将 Type 枚举转换为字符串
//   3. is_supported()    - 检查某个类型是否被当前版本支持
//   4. quantized_size()  - 计算给定形状和量化类型的存储空间大小
// =============================================================================

// 量化抽象层的实现

#include "quant.h"

namespace tinyqwen {
namespace quantization {

    // 将类型名称字符串解析为 Type 枚举
    // 支持别名：f32/fp32、f16/fp16、i4/int4
    // 参数：
    //   str - 类型名称字符串（不区分别名，但区分大小写）
    //   out - 输出参数，解析成功时写入对应的 Type 枚举值
    // 返回：
    //   true  解析成功
    //   false 不认识的类型名称
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

    // 将 Type 枚举转换为字符串名称
    // 返回的是规范名称（f32/f16/i4），不是别名
    const char* type_name(Type type) {
        switch (type) {
            case Type::kF32: return "f32";
            case Type::kF16: return "f16";
            case Type::kI4:  return "i4";
            default:         return "unknown";
        }
    }

    // 检查某个数据类型是否被当前版本的 tinyqwen 支持
    // 当前支持：FP32（全精度浮点）、FP16（半精度浮点）、INT4（4-bit 整数）
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

    // 计算给定形状和量化类型的存储空间大小（字节数）
    // 参数：
    //   type       - 数据类型（FP32/FP16/INT4）
    //   rows       - 矩阵行数（通常是输出维度 out_dim）
    //   cols       - 矩阵列数（通常是输入维度 in_dim）
    //   group_size - INT4 量化组大小（FP32/FP16 忽略此参数）
    // 返回：
    //   存储该矩阵所需的字节数
    //
    // INT4 存储布局：
    //   每 group_size 个元素共享一个 scale(fp16) 和一个 zero(fp16)
    //   权重数据：每 2 个 4-bit 值打包为 1 字节
    //   量化参数：每行 groups_per_row 组 × 2 参数(scale+zero) × 2 字节(fp16)
    size_t quantized_size(Type type, int rows, int cols, int group_size) {
        switch (type) {
            case Type::kF32:
                // FP32: 每元素 4 字节
                return static_cast<size_t>(rows) * cols * 4;
            case Type::kF16:
                // FP16: 每元素 2 字节
                return static_cast<size_t>(rows) * cols * 2;
            case Type::kI4: {
                // INT4: 每组 1 byte（2 个 4-bit 值） + scale (fp16) + zero (fp16)
                const int num_groups = (cols + group_size - 1) / group_size;  // 向上取整
                const size_t data_bytes = static_cast<size_t>(rows) * ((cols + 1) / 2);  // 权重数据
                const size_t param_bytes = static_cast<size_t>(rows) * num_groups * 2 * 2;  // 量化参数
                return data_bytes + param_bytes;
            }
            default:
                return 0;
        }
    }

} // namespace quantization
} // namespace tinyqwen
