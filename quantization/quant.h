#pragma once

// 量化抽象层：定义量化类型的元数据和工具函数
//
// 当前支持的量化算法：
//   - FP32：无量化
//   - FP16：半精度浮点
//   - INT4：非对称 uint4，per-group scale+zero（RTN / HQQ）
//
// 未来扩展：
//   - INT8：非对称 uint8，per-group scale+zero
//   - GPTQ：group-wise quantization with error compensation
//   - AWQ：activation-aware quantization

#include <cstdint>
#include <string>

namespace tinyqwen {
namespace quantization {

    // 量化类型（与 runtime/backend.h 的 QuantType 对应）
    enum class Type {
        kF32 = 0,
        kF16 = 1,
        kI4 = 2,
        // 未来扩展：kI8, kGPTQ, kAWQ, ...
    };

    // 量化元数据
    struct Metadata {
        Type type;
        int group_size;          // group size（INT4/INT8 时 > 0）
        bool symmetric;          // 是否对称量化
        std::string algorithm;   // 算法名（"rtn" / "hqq" / "gptq" / ...）
    };

    // 解析量化类型（从字符串）
    bool parse_type(const std::string& str, Type* out);

    // 量化类型转字符串
    const char* type_name(Type type);

    // 检查量化类型是否支持
    bool is_supported(Type type);

    // 计算量化后的数据大小（字节）
    size_t quantized_size(Type type, int rows, int cols, int group_size);

} // namespace quantization
} // namespace tinyqwen
