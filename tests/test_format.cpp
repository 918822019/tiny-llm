#include "test_framework.h"

#include <cstring>

#include "tiny_format.h"

using namespace tinyqwen;

TEST (format_header_layout) {
    EXPECT_EQ(sizeof(TinyHeader), (size_t) 192);
    EXPECT_EQ(sizeof(TensorEntry), (size_t) 120);
    EXPECT_EQ(sizeof(TinyHeaderV2Ext), (size_t) 64);
    EXPECT_EQ(std::memcmp(kMagic, "TINYQWEN", 8), 0);
    EXPECT_EQ(kFormatVersion, 2u);
    EXPECT_EQ(kFormatVersionMin, 1u);
    EXPECT_EQ(kAlignment, (size_t) 64);
    EXPECT_EQ((int) Dtype::kF32, 0);
    EXPECT_EQ((int) ModelType::kQwen2, 0);
    EXPECT_EQ((int) ModelType::kQwen35, 1);
}

TEST (format_dtype_sizes) {
    EXPECT_EQ(dtype_size(Dtype::kF32), (size_t) 4);
    EXPECT_EQ(dtype_size(Dtype::kF16), (size_t) 2);
    EXPECT_EQ(dtype_size(Dtype::kI8), (size_t) 1);
    EXPECT_EQ(dtype_size(Dtype::kI4), (size_t) 0); // 亚字节类型，布局由 kernel 决定
    EXPECT_EQ(std::strcmp(dtype_name(Dtype::kF32), "f32"), 0);
}

TEST (format_field_offsets) {
    // 这些断言锁死磁盘布局；必须与 tools/export_qwen_to_tiny.py 保持同步。
    EXPECT_EQ(offsetof(TinyHeader, magic), (size_t) 0);
    EXPECT_EQ(offsetof(TinyHeader, version), (size_t) 8);
    EXPECT_EQ(offsetof(TinyHeader, n_layers), (size_t) 16);
    EXPECT_EQ(offsetof(TinyHeader, rms_norm_eps), (size_t) 56);
    EXPECT_EQ(offsetof(TinyHeader, tensor_count), (size_t) 64);
    EXPECT_EQ(offsetof(TinyHeader, reserved), (size_t) 96);
    EXPECT_EQ(offsetof(TensorEntry, dtype), (size_t) 64);
    EXPECT_EQ(offsetof(TensorEntry, ndim), (size_t) 68);
    EXPECT_EQ(offsetof(TensorEntry, shape), (size_t) 72);
    EXPECT_EQ(offsetof(TensorEntry, offset), (size_t) 104);
    EXPECT_EQ(offsetof(TensorEntry, nbytes), (size_t) 112);
}
