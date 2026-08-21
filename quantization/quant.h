// ============================================================================
// quant.h — 量化抽象层头文件
// ============================================================================
// 本文件定义量化类型的枚举、元数据结构及工具函数接口。
// 是 tinyqwen 量化子系统的公共 API，供 runtime、kernel、导出工具共同使用。
//
// 当前支持的量化算法：
//   - FP32（kF32）：无量化，原始浮点权重
//   - FP16（kF16）：半精度浮点，权重存储减半
//   - INT4（kI4）：非对称 uint4 量化，per-group scale+zero（RTN / HQQ）
//     每个 group 有独立的 fp16 scale 和 zero，反量化公式：w = (q - zero) * scale
//
// 未来扩展预留：
//   - INT8（kI8）：非对称 uint8，per-group scale+zero
//   - GPTQ：group-wise quantization with error compensation
//   - AWQ：activation-aware quantization
//
// 与 runtime/backend.h 的 QuantType 枚举保持对应关系。
// ============================================================================

#pragma once  // 防止头文件被重复包含

#include <cstdint>   // uint8_t, uint16_t 等固定宽度整数类型
#include <string>    // std::string（algorithm 字段）

namespace tinyqwen {
namespace quantization {

    // =========================================================================
    // enum class Type — 量化类型枚举
    // =========================================================================
    // 与 runtime/backend.h 的 QuantType 一一对应。
    // 用于模型文件格式标识、dispatch 路径选择、内存布局计算等场景。
    enum class Type {
        kF32 = 0,   // FP32：无量化，权重以 float32 存储
        kF16 = 1,   // FP16：半精度浮点，权重以 float16 存储
        kI4 = 2,    // INT4：非对称 uint4 量化，per-group scale(fp16)+zero(fp16)
        // 未来扩展：kI8, kGPTQ, kAWQ, ...
    };

    // =========================================================================
    // struct Metadata — 量化元数据
    // =========================================================================
    // 描述一个量化配置的完整参数集。存储在模型文件的头部，加载时读取。
    struct Metadata {
        Type type;               // 量化类型（FP32/FP16/INT4/...）
        int group_size;          // 分组大小（INT4/INT8 时 > 0，如 64 或 128）
                                 // 每组共享一个 scale 和 zero 参数
        bool symmetric;          // 是否对称量化（true 时 zero=0，只存 scale）
        std::string algorithm;   // 量化算法名称字符串（"rtn" / "hqq" / "gptq" / ...）
                                 // 用于日志记录和调试，不影响运行时逻辑
    };

    // =========================================================================
    // parse_type() — 从字符串解析量化类型
    // =========================================================================
    // 参数：
    //   str — 量化类型字符串（如 "f32", "fp16", "i4"）
    //   out — 输出参数，解析成功时写入对应的 Type 值
    // 返回：true = 解析成功，false = 无法识别的字符串
    bool parse_type(const std::string& str, Type* out);

    // =========================================================================
    // type_name() — 量化类型转可读字符串
    // =========================================================================
    // 参数：type — 量化类型枚举值
    // 返回：对应的 C 字符串（如 "f32", "f16", "i4"），用于日志和错误信息
    const char* type_name(Type type);

    // =========================================================================
    // is_supported() — 检查量化类型是否在当前构建中支持
    // =========================================================================
    // 参数：type — 待检查的量化类型
    // 返回：true = 支持（有对应的 kernel 实现），false = 不支持
    bool is_supported(Type type);

    // =========================================================================
    // quantized_size() — 计算量化后的数据大小（字节）
    // =========================================================================
    // 参数：
    //   type       — 量化类型
    //   rows       — 矩阵行数（output dimension）
    //   cols       — 矩阵列数（input dimension）
    //   group_size — 分组大小（INT4/INT8 时使用）
    // 返回：量化后权重的总字节数（含 scale/zero 头部开销）
    // 用途：预分配内存、校验文件大小、估算内存占用
    size_t quantized_size(Type type, int rows, int cols, int group_size);

} // namespace quantization
} // namespace tinyqwen
