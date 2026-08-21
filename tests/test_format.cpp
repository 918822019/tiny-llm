// ============================================================================
// test_format.cpp — 模型文件格式（TinyHeader / TensorEntry）单元测试
// ============================================================================
// 本文件测试 tinyqwen 模型文件格式的二进制布局，确保磁盘结构定义与
// Python 导出工具（tools/export_qwen_to_tiny.py）保持一致。
//
// 测试覆盖：
//   1. 结构体大小验证（sizeof 检查）
//   2. 魔术数和版本常量验证
//   3. 数据类型大小验证（dtype_size）
//   4. 字段偏移量验证（offsetof 检查，锁死磁盘布局）
//
// 这些测试直接关系到文件格式的向前/向后兼容性：如果结构体大小或字段偏移
// 发生变化，将会导致模型文件无法正确加载，测试会立即失败以提供早期预警。
// ============================================================================

#include "test_framework.h"

#include <cstring>

#include "tiny_format.h"

using namespace tinyqwen;

// =============================================================================
// format_header_layout — 结构体大小和常量验证
// =============================================================================
// 验证 TinyHeader（192 字节）、TensorEntry（120 字节）和 TinyHeaderV2Ext
// （64 字节）的大小，以及魔数、版本号、对齐值、枚举常量等基础常量。
// 这些值必须与 Python 导出工具中的定义完全一致，否则会导致文件格式不兼容。
TEST (format_header_layout) {
    // 验证各结构体的 sizeof，锁死磁盘布局
    EXPECT_EQ(sizeof(TinyHeader), (size_t) 192);
    EXPECT_EQ(sizeof(TensorEntry), (size_t) 120);
    EXPECT_EQ(sizeof(TinyHeaderV2Ext), (size_t) 64);
    // 验证魔数字符串 "TINYQWEN"（8 字节）
    EXPECT_EQ(std::memcmp(kMagic, "TINYQWEN", 8), 0);
    // 验证版本号常量
    EXPECT_EQ(kFormatVersion, 2u);
    EXPECT_EQ(kFormatVersionMin, 1u);
    // 验证对齐要求：所有数据段偏移 64 字节对齐
    EXPECT_EQ(kAlignment, (size_t) 64);
    // 验证枚举类型的底层整数值
    EXPECT_EQ((int) Dtype::kF32, 0);
    EXPECT_EQ((int) ModelType::kQwen2, 0);
    EXPECT_EQ((int) ModelType::kQwen35, 1);
}

// =============================================================================
// format_dtype_sizes — 数据类型大小验证
// =============================================================================
// 验证 dtype_size() 函数返回的每种数据类型字节数。
// 特别注意 INT4 是亚字节类型（两个元素打包在一个字节中），
// 所以 dtype_size 返回 0 表示"无法用每元素字节数描述"。
TEST (format_dtype_sizes) {
    EXPECT_EQ(dtype_size(Dtype::kF32), (size_t) 4);  // float32: 4 字节
    EXPECT_EQ(dtype_size(Dtype::kF16), (size_t) 2);  // float16: 2 字节
    EXPECT_EQ(dtype_size(Dtype::kI8), (size_t) 1);   // int8:    1 字节
    EXPECT_EQ(dtype_size(Dtype::kI4), (size_t) 0);   // int4:    亚字节类型，布局由 kernel 决定
    EXPECT_EQ(std::strcmp(dtype_name(Dtype::kF32), "f32"), 0); // 名称验证
}

// =============================================================================
// format_field_offsets — 字段偏移量验证
// =============================================================================
// 使用 offsetof 宏验证 TinyHeader 和 TensorEntry 结构体中各字段的偏移量。
// 这些断言锁定了磁盘布局，必须与 tools/export_qwen_to_tiny.py 中的
// struct.pack 格式字符串保持同步。
// 如果任何人修改了结构体定义（如添加/删除/重排字段），这些测试会立即失败，
// 从而防止引入不兼容的格式变更。
TEST (format_field_offsets) {
    // 这些断言锁死磁盘布局；必须与 tools/export_qwen_to_tiny.py 保持同步。

    // TinyHeader 字段偏移量验证
    EXPECT_EQ(offsetof(TinyHeader, magic), (size_t) 0);        // 魔数：8 字节
    EXPECT_EQ(offsetof(TinyHeader, version), (size_t) 8);      // 版本号：8 字节
    EXPECT_EQ(offsetof(TinyHeader, n_layers), (size_t) 16);    // 层数：8 字节
    // ... 中间字段略过 ...
    EXPECT_EQ(offsetof(TinyHeader, rms_norm_eps), (size_t) 56); // RMSNorm epsilon：8 字节
    EXPECT_EQ(offsetof(TinyHeader, tensor_count), (size_t) 64); // 张量数量：8 字节
    EXPECT_EQ(offsetof(TinyHeader, reserved), (size_t) 96);     // 预留扩展区：96 字节

    // TensorEntry 字段偏移量验证
    EXPECT_EQ(offsetof(TensorEntry, dtype), (size_t) 64);      // 数据类型：4 字节
    EXPECT_EQ(offsetof(TensorEntry, ndim), (size_t) 68);       // 维度数：4 字节
    EXPECT_EQ(offsetof(TensorEntry, shape), (size_t) 72);      // 形状数组：32 字节
    EXPECT_EQ(offsetof(TensorEntry, offset), (size_t) 104);    // 文件偏移：8 字节
    EXPECT_EQ(offsetof(TensorEntry, nbytes), (size_t) 112);    // 字节数：8 字节
}