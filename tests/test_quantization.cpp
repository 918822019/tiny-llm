// 量化抽象层的单元测试

#include "test_framework.h"

#include <cstring>

#include "../quantization/quant.h"

using namespace tinyqwen::quantization;

TEST (quant_parse_type) {
    Type t;
    EXPECT_TRUE(parse_type("f32", &t) && t == Type::kF32);
    EXPECT_TRUE(parse_type("fp32", &t) && t == Type::kF32);
    EXPECT_TRUE(parse_type("f16", &t) && t == Type::kF16);
    EXPECT_TRUE(parse_type("fp16", &t) && t == Type::kF16);
    EXPECT_TRUE(parse_type("i4", &t) && t == Type::kI4);
    EXPECT_TRUE(parse_type("int4", &t) && t == Type::kI4);
    EXPECT_TRUE(!parse_type("unknown", &t));
}

TEST (quant_type_name) {
    EXPECT_TRUE(std::strcmp(type_name(Type::kF32), "f32") == 0);
    EXPECT_TRUE(std::strcmp(type_name(Type::kF16), "f16") == 0);
    EXPECT_TRUE(std::strcmp(type_name(Type::kI4), "i4") == 0);
}

TEST (quant_is_supported) {
    EXPECT_TRUE(is_supported(Type::kF32));
    EXPECT_TRUE(is_supported(Type::kF16));
    EXPECT_TRUE(is_supported(Type::kI4));
}

TEST (quant_quantized_size) {
    // f32: 128 * 256 * 4 = 131072 bytes
    EXPECT_EQ(quantized_size(Type::kF32, 128, 256, 0), 131072);

    // f16: 128 * 256 * 2 = 65536 bytes
    EXPECT_EQ(quantized_size(Type::kF16, 128, 256, 0), 65536);

    // i4: data = 128 * 128 = 16384 bytes
    //     params = 128 * 4 * 2 * 2 = 2048 bytes (group_size=64, 256/64=4 groups)
    //     total = 16384 + 2048 = 18432 bytes
    size_t i4_size = quantized_size(Type::kI4, 128, 256, 64);
    EXPECT_TRUE(i4_size > 16384 && i4_size < 32768);
}
