#pragma once

// 本文件定义 .tqwen 权重文件的二进制格式，是 C++ 与 Python 共享的"契约"。
// 第一次看请先读 docs/infra_primer.md 的第 4、5 节（对齐 / 字节序 / padding）。

#include <cstddef>
#include <cstdint>

namespace tinyqwen {
    // ---------------------------------------------------------------------------
    // tinyqwen 扁平二进制格式（".tqwen"），版本 1。
    //
    // 为什么要有自己的格式？
    //   原始的 safetensors 带 JSON 头、可能分多个分片，运行时解析它既复杂又慢。
    //   我们提前用 Python 把权重"压平"成一个结构极简、可直接读的文件：
    //   加载端只需要按固定偏移读，不用做任何文本解析。
    //
    // 文件布局（从上到下依次存放）：
    //   [ TinyHeader                     ]  192 字节，文件头，见下面的结构体
    //   [ TensorEntry * tensor_count     ]  "tensor 表"，每项 120 字节，描述一个 tensor
    //   [ 64 字节对齐的填充              ]  用 0 补齐，让数据区起点 64B 对齐
    //   [ tensor 数据区                  ]  真正的权重数字，每个 tensor 起点都 64B 对齐
    //
    // 两条硬性约定：
    //   - 所有整数用**小端**存放（ARM64/x86 都是小端，见 primer 第 5 节）；
    //   - 所有 offset 都是**相对文件开头的绝对偏移**，不是相对当前位置。
    //
    // 这个头文件是唯一契约，同时约束：
    //   - C++ 加载端 runtime/model_loader.cpp
    //   - Python 导出端 tools/export_qwen_to_tiny.py
    // 任何修改都必须同时 bump kFormatVersion 并改 exporter，否则两边对不上。
    // 完整规范见 docs/weight_format.md。
    // ---------------------------------------------------------------------------

    // 魔数（magic）：文件开头 8 个字节固定是 "TINYQWEN"。
    // 加载时先看这 8 个字节，就能判断"这是不是我们的文件"，防止把别的文件
    // 当成权重读进来。
    inline constexpr char kMagic[8] = {'T', 'I', 'N', 'Y', 'Q', 'W', 'E', 'N'};

    // 格式版本号。将来改了布局就 +1，老 loader 遇到新版本会直接拒绝，
    // 而不是硬着头皮读错。
    inline constexpr uint32_t kFormatVersion = 1;

    // 对齐粒度：每个 tensor 数据的起始偏移都要是 64 的倍数。
    // 64 = 常见 CPU 缓存行大小，对齐后读取缓存友好（见 primer 第 4 节）。
    inline constexpr size_t kAlignment = 64;

    // 数据类型编号。v1 只用 f32，其余是为将来量化预留的。
    enum class Dtype : uint32_t {
        kF32 = 0, // 32 位浮点，v1 唯一支持的类型
        kF16 = 1, // 预留：16 位浮点
        kI8 = 2, // 预留：weight-only INT8 量化
        kI4 = 3, // 预留：打包 INT4 / KronQ（亚字节，需要专门布局）
    };

    // 每个元素占多少字节；未知/亚字节类型返回 0。
    size_t dtype_size(Dtype dtype);

    const char *dtype_name(Dtype dtype);

    // 文件头，固定 192 字节，放在文件最前面。
    // 注意字段顺序是精心安排的：u64 字段都落在 8 的倍数偏移上，
    // 这样编译器不会插入 padding，内存布局 == 磁盘布局（见 primer 第 5 节）。
    struct TinyHeader {
        char magic[8]; // 魔数，必须等于 kMagic
        uint32_t version; // 格式版本，必须等于 kFormatVersion
        uint32_t dtype; // 所有 tensor 的默认 dtype（v1: 0 = f32）
        // --- 模型配置（描述这是一个什么形状的模型）---
        uint32_t n_layers; // transformer 层数
        uint32_t hidden_size; // 隐藏层宽度（每个 token 的向量维度）
        uint32_t intermediate_size; // FFN 中间层宽度
        uint32_t n_heads; // query 注意力头数
        uint32_t n_kv_heads; // key/value 头数（GQA，通常小于 n_heads）
        uint32_t head_dim; // 每个头的维度
        uint32_t vocab_size; // 词表大小
        uint32_t max_seq_len; // 训练时支持的最大序列长度
        uint32_t tied_embeddings; // 1 => lm_head 与 embed_tokens 共享同一份权重
        uint32_t reserved_u32; // 预留，必须填 0
        float rms_norm_eps; // RMSNorm 里防止除零的小常数
        float rope_theta; // RoPE 位置编码的底数 theta
        // --- 表区 / 数据区的位置 ---
        uint64_t tensor_count; // tensor 表里有多少项
        uint64_t tensor_table_offset; // tensor 表的起始偏移（v1 固定 = 192）
        uint64_t data_offset; // 第一个 tensor 数据的偏移，% 64 == 0
        uint64_t total_bytes; // 整个文件的字节数（加载时用来校验）
        char reserved[96]; // 预留空间，必须全 0，方便将来扩展
    };

    // tensor 名字最长 64 字节。
    inline constexpr size_t kMaxTensorName = 64;

    // tensor 表的一项，描述"某一个 tensor 在哪、多大、什么形状"。
    // 注意：这里不存数据本身，只存"指向数据区的元信息"。
    struct TensorEntry {
        char name[kMaxTensorName]; // 名字，用 NUL 补齐；占满 64 字符时无终止符
        uint32_t dtype; // 数据类型，见 Dtype
        uint32_t ndim; // 维度数 1..4，shape 里后面对应的位才有效
        uint64_t shape[4]; // 形状；shape[ndim..3] 必须为 0
        uint64_t offset; // 数据在文件里的绝对偏移，% 64 == 0
        uint64_t nbytes; // 数据字节数 = 元素个数 * 每元素字节数
    };

    // 编译期断言：把结构体大小"锁死"。
    // 一旦有人加字段导致编译器插入 padding、大小变了，编译立刻报错，
    // 避免"内存布局和磁盘布局悄悄不一致"这种极难排查的 bug。
    static_assert(sizeof(TinyHeader) == 192, "TinyHeader must be exactly 192 bytes");
    static_assert(sizeof(TensorEntry) == 120, "TensorEntry must be exactly 120 bytes");
    static_assert(offsetof(TinyHeader, tensor_count) % 8 == 0,
                  "u64 fields must stay 8-byte aligned");
    static_assert(offsetof(TensorEntry, shape) % 8 == 0,
                  "u64 fields must stay 8-byte aligned");
} // namespace tinyqwen
