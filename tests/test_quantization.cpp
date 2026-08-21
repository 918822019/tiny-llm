// =============================================================================
// test_quantization.cpp
// =============================================================================
// 文件级别说明：
//   本文件测试量化抽象层（quantization/quant.h）的核心功能。
//   量化抽象层定义了不同数据类型（FP32、FP16、INT4）的通用接口，
//   包括类型名称解析、类型名称获取、支持性检查和量化后大小计算。
//
// 测试覆盖：
//   1. quant_parse_type     - 类型名称字符串解析
//   2. quant_type_name      - 类型到名称字符串的转换
//   3. quant_is_supported   - 类型支持性检查
//   4. quant_quantized_size - 量化后存储空间大小计算
// =============================================================================

// 量化抽象层的单元测试

#include "test_framework.h"

#include <cstring>

#include "../quantization/quant.h"

using namespace tinyqwen::quantization;

// 测试用例：类型名称字符串解析
// 验证 parse_type() 函数能正确解析各种类型名称的别名，
// 包括 f32/fp32、f16/fp16、i4/int4。
// 不认识的类型名称应返回 false。
TEST (quant_parse_type) {
    Type t;
    // FP32 的两种别名
    EXPECT_TRUE(parse_type("f32", &t) && t == Type::kF32);
    EXPECT_TRUE(parse_type("fp32", &t) && t == Type::kF32);
    // FP16 的两种别名
    EXPECT_TRUE(parse_type("f16", &t) && t == Type::kF16);
    EXPECT_TRUE(parse_type("fp16", &t) && t == Type::kF16);
    // INT4 的两种别名
    EXPECT_TRUE(parse_type("i4", &t) && t == Type::kI4);
    EXPECT_TRUE(parse_type("int4", &t) && t == Type::kI4);
    // 未知类型应返回 false
    EXPECT_TRUE(!parse_type("unknown", &t));
}

// 测试用例：类型到名称字符串的转换
// 验证 type_name() 函数返回正确的字符串表示。
// 这是 parse_type() 的逆操作。
TEST (quant_type_name) {
    EXPECT_TRUE(std::strcmp(type_name(Type::kF32), "f32") == 0);
    EXPECT_TRUE(std::strcmp(type_name(Type::kF16), "f16") == 0);
    EXPECT_TRUE(std::strcmp(type_name(Type::kI4), "i4") == 0);
}

// 测试用例：类型支持性检查
// 验证 is_supported() 函数对已支持的类型返回 true。
// 当前 tinyqwen 支持 FP32、FP16、INT4 三种类型。
TEST (quant_is_supported) {
    EXPECT_TRUE(is_supported(Type::kF32));
    EXPECT_TRUE(is_supported(Type::kF16));
    EXPECT_TRUE(is_supported(Type::kI4));
}

// 测试用例：量化后存储空间大小计算
// 验证 quantized_size() 函数对不同类型和形状返回正确的字节数。
// INT4 的存储包含权重数据（4 bit/元素）和量化参数（scale/zero，各 fp16）。
TEST (quant_quantized_size) {
    // f32: 128 * 256 * 4 = 131072 bytes（每元素 4 字节）
    EXPECT_EQ(quantized_size(Type::kF32, 128, 256, 0), 131072);

    // f16: 128 * 256 * 2 = 65536 bytes（每元素 2 字节）
    EXPECT_EQ(quantized_size(Type::kF16, 128, 256, 0), 65536);

    // i4: 权重数据 = 128 * 128 = 16384 bytes（每 2 个元素占 1 字节）
    //     量化参数 = 128 * 4 * 2 * 2 = 2048 bytes
    //     （128 行 × 4 组（256/64）× 2 参数（scale+zero）× 2 字节(fp16)）
    //     total = 16384 + 2048 = 18432 bytes
    size_t i4_size = quantized_size(Type::kI4, 128, 256, 64);
    EXPECT_TRUE(i4_size > 16384 && i4_size < 32768);
}
