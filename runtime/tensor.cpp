// ============================================================================
// tensor.cpp — 张量基础类型定义
// ============================================================================
// 本文件定义了 tensor 系统中与数据类型（Dtype）和 TensorView（张量视图）
// 相关的基础设施。
//
// Dtype 枚举支持四种数据类型：
//   - kF32: 32 位浮点数（IEEE 754 single precision），每个元素占 4 字节
//   - kF16: 16 位浮点数（IEEE 754 half precision），每个元素占 2 字节
//   - kI8:  8 位有符号整数，每个元素占 1 字节
//   - kI4:  4 位有符号整数（亚字节），两个元素打包进 1 字节，无法用"每元素
//          字节数"描述，所以 dtype_size 返回 0
//
// TensorView 是零拷贝的轻量级张量视图：它不拥有数据，只持有指向文件内存中
// 数据的指针（const uint8_t* data），加上 shape/ndim/dtype 等元信息。
// 这使得模型加载后可以直接从文件内存中读取权重，无需额外拷贝。
// ============================================================================

#include "tensor.h"

#include <cstdio>   // 标准输入输出（fprintf, stderr）
#include <cstdlib>  // 标准库（abort）

namespace tinyqwen {
    // =========================================================================
    // dtype_size() — 获取每种数据类型的每元素字节数
    // =========================================================================
    // 参数：
    //   dtype — 数据类型枚举值
    // 返回值：每元素占用的字节数
    // 说明：i4 是亚字节类型（一个字节装两个数），没法用"每元素字节数"描述，
    //       所以返回 0，由具体 kernel 特殊处理。
    size_t dtype_size(Dtype dtype) {
        switch (dtype) {
            case Dtype::kF32: return 4;  // float32: 4 字节
            case Dtype::kF16: return 2;  // float16: 2 字节
            case Dtype::kI8:  return 1;  // int8:    1 字节
            case Dtype::kI4:  return 0;  // int4:    亚字节，无法简单描述
            case Dtype::kVQ2: return 0;  // vq2:     每权重 1B 索引 + per-tensor 码本，整体大小见 vq2_tensor_bytes
            case Dtype::kGPTQ4: return 0; // gptq4:  列主序 int32 打包 + 分离 scales/qzeros/g_idx，整体大小见 gptq_tensor_bytes
        }
        return 0; // 未知类型，保守返回 0
    }

    // =========================================================================
    // dtype_name() — 获取数据类型的可读名称
    // =========================================================================
    // 参数：
    //   dtype — 数据类型枚举值
    // 返回值：类型名称字符串（如 "f32", "f16", "i8", "i4"）
    // 说明：用于日志输出和错误信息中显示数据类型。
    const char *dtype_name(Dtype dtype) {
        switch (dtype) {
            case Dtype::kF32: return "f32";
            case Dtype::kF16: return "f16";
            case Dtype::kI8:  return "i8";
            case Dtype::kI4:  return "i4";
            case Dtype::kVQ2: return "vq2";
            case Dtype::kGPTQ4: return "gptq4";
        }
        return "?"; // 未知类型
    }

    // =========================================================================
    // TensorView::numel() — 计算张量中元素的总个数
    // =========================================================================
    // 返回值：所有维度乘积 = shape[0] * shape[1] * ... * shape[ndim-1]
    // 说明：对于 INT4 类型，返回的是逻辑元素个数（而非物理字节数）。
    uint64_t TensorView::numel() const {
        uint64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i]; // 逐维连乘
        return n;
    }

    // =========================================================================
    // TensorView::f32() — 将 data 指针以 float32 视角解释
    // =========================================================================
    // 返回值：指向 float32 数据的 const 指针
    // 说明：data 字段是 uint8_t*（按字节寻址），但权重数据要按 float 读取，
    //       所以需要 reinterpret_cast 成 float*。这要求 data 地址对 float 的
    //       对齐要求满足——我们的 64B 对齐保证了这一点。如果 dtype 不是 kF32，
    //       会触发 abort（fail fast，防止静默算错）。
    const float *TensorView::f32() const {
        // 类型检查：只有 kF32 类型的张量才能调用 f32()
        if (dtype != Dtype::kF32) {
            std::fprintf(stderr, "tinyqwen: tensor '%s' has dtype %s, f32 expected\n",
                         name.c_str(), dtype_name(dtype));
            std::abort(); // 类型不匹配，立即终止
        }
        // 将字节指针 reinterpret 为 float 指针（零拷贝，不分配新内存）
        return reinterpret_cast<const float *>(data);
    }
} // namespace tinyqwen