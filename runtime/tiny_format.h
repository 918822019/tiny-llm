#pragma once

// ============================================================================
// 文件: tiny_format.h
// 作用: 定义 .tqwen 权重文件的二进制格式 —— C++ 与 Python 共享的"契约"
//
// 为什么要有自己的格式?
//   原始的 safetensors 带 JSON 头、可能分多个分片，运行时解析它既复杂又慢。
//   我们提前用 Python 把权重"压平"成一个结构极简、可直接读的文件:
//   加载端只需要按固定偏移读，不用做任何文本解析。
//
// 文件布局（从上到下依次存放）:
//   [ TinyHeader                     ]  192 字节，文件头
//   [ TensorEntry * tensor_count     ]  "tensor 表"，每项 120 字节
//   [ 64 字节对齐的填充              ]  用 0 补齐，让数据区起点 64B 对齐
//   [ tensor 数据区                  ]  真正的权重数字，每个 tensor 起点 64B 对齐
//
// 两条硬性约定:
//   - 所有整数用小端存放（ARM64/x86 都是小端）
//   - 所有 offset 都是相对文件开头的绝对偏移
//
// 本头文件是唯一契约，同时约束:
//   - C++ 加载端: runtime/model_loader.cpp
//   - Python 导出端: tools/export_qwen_to_tiny.py
// 任何修改都必须同时 bump kFormatVersion 并改 exporter，否则两边对不上。
//
// 完整规范: docs/weight_format.md
// 前置阅读: docs/infra_primer.md 第 4、5 节（对齐 / 字节序 / padding）
// ============================================================================

#include <cstddef>
#include <cstdint>

namespace tinyqwen {
    // =========================================================================
    // 文件格式常量
    // =========================================================================

    // 魔数: 文件开头 8 个字节固定是 "TINYQWEN"
    // 加载时先看这 8 个字节判断"这是不是我们的文件"，防止误读其他文件
    inline constexpr char kMagic[8] = {'T', 'I', 'N', 'Y', 'Q', 'W', 'E', 'N'};

    // 格式版本号
    //   v1: Qwen2.x 单一架构（所有层同构 full attention + SwiGLU）
    //   v2: header 布局不变（仍是 192 字节），reserved[96] 按 TinyHeaderV2Ext 解读
    // 将来改了布局就 +1，老 loader 遇到新版本会直接拒绝
    inline constexpr uint32_t kFormatVersion = 2;

    // loader 接受的最低版本（v1 文件向后兼容）
    inline constexpr uint32_t kFormatVersionMin = 1;

    // 对齐粒度: 每个 tensor 数据的起始偏移都要是 64 的倍数
    // 64 = 常见 CPU 缓存行大小，对齐后读取缓存友好
    inline constexpr size_t kAlignment = 64;

    // =========================================================================
    // INT4 量化 packing 布局常量
    //
    // 非对称 uint4 [0,15]，per-group (128 元素) 有 scale + zero_point（均 fp16）
    // 每组内存布局: [scale_fp16(2B) | zero_fp16(2B) | packed_uint4(64B)] = 68B
    // 反量化: float_val = (uint4_val - zero) * scale
    // 低 nibble 在前: byte & 0x0F = 偶数下标元素，byte >> 4 = 奇数下标元素
    // =========================================================================
    inline constexpr int kI4DefaultGroupSize = 128;   // 每组 128 个元素
    inline constexpr int kI4GroupHeaderBytes = 4;     // 每组头部: scale(2) + zero(2)
    inline constexpr int kI4GroupDataBytes = 64;      // 每组数据: 128 nibbles 打包
    inline constexpr int kI4GroupTotalBytes = 68;     // 每组总字节: header + data

    // 给定 group_size，计算单行的字节数
    // n_groups = ceil(in_dim / group_size)，每组 header + data
    inline constexpr size_t i4_row_bytes(int in_dim, int group_size) {
        const int n_groups = (in_dim + group_size - 1) / group_size; // 向上取整
        const int data_bytes = group_size / 2;                       // 每组打包数据字节数
        return static_cast<size_t>(n_groups) * (kI4GroupHeaderBytes + data_bytes);
    }

    // =========================================================================
    // VQ2（2-bit 向量量化）布局常量
    //
    // 每个被量化的权重张量 = 一个 [out_dim, in_dim] 矩阵 + 一份码本。
    // 码本 in-band 存放在张量数据区最前面（与 i4 把 scale/zero 内联进权重同思路）：
    //   [ codebook: [K, d] 个 fp16 = kVQ2CodebookBytes 字节 ]
    //   [ indices:  out_dim × (in_dim/d) 个 uint8，行主序，每 d 个连续权重 1 字节 ]
    // 块向量量化：每 d(=4) 个连续权重 = 一个 d 维向量，用 1 个索引编码；
    // 反量化：取 codebook[index] 这个 d 维向量（fp16 → fp32），纯查表、零算术。
    // 码率 = log2(K)/d = log2(256)/4 = 2 bit/权重。
    // 2.0bit 的"部署洁净点"：每块恰 1 字节、字节对齐、免 bit-pack。
    // =========================================================================
    inline constexpr int kVQ2BlockDim = 4;                 // 块大小 d：每个索引编码 d 个连续权重
    inline constexpr int kVQ2CodebookEntries = 256;        // 码本条目数 K = 256
    // 码本 [K, d] fp16 字节数 = 256 * 4 * 2 = 2048B
    inline constexpr int kVQ2CodebookBytes = kVQ2CodebookEntries * kVQ2BlockDim * 2;

    // 给定形状，计算整个 VQ2 张量的字节数（码本 + 索引）
    // 块向量量化：每 d(=4) 个连续权重共用 1 个字节索引 → 码率 = log2(K)/d = 2 bit/权重
    // 索引区 = rows * (cols / d) 字节；要求 cols 能被 d 整除
    // nbytes = kVQ2CodebookBytes + rows * (cols / kVQ2BlockDim)
    inline constexpr size_t vq2_tensor_bytes(int rows, int cols) {
        return kVQ2CodebookBytes +
               static_cast<size_t>(rows) * (static_cast<size_t>(cols) / kVQ2BlockDim);
    }

    // 模型架构族枚举。决定 forward 走哪条路径、按什么名字绑定权重
    enum class ModelType : uint32_t {
        kQwen2 = 0,  // Qwen2 / Qwen2.5: 所有层同构（full attention + SwiGLU）
        kQwen35 = 1, // Qwen3.5 混合架构: Gated DeltaNet + full attention 3:1 交替
    };

    // 数据类型枚举。v1 只用 f32，其余是为将来量化预留的
    enum class Dtype : uint32_t {
        kF32 = 0, // 32 位浮点（IEEE 754 single precision）
        kF16 = 1, // 16 位浮点（IEEE 754 half precision），weight-only 半精度
        kI8 = 2,  // 预留: weight-only INT8 量化
        kI4 = 3,  // 打包 INT4 量化（亚字节，需要专门布局）
        kVQ2 = 4, // 2-bit 向量量化：每权重 1 字节 uint8 码本索引 + per-tensor 码本
    };

    // 返回每种 dtype 每元素占多少字节；未知/亚字节类型返回 0
    size_t dtype_size(Dtype dtype);

    // 返回 dtype 的可读名称，如 "f32"、"f16"
    const char *dtype_name(Dtype dtype);

    // -------------------------------------------------------------------------
    // TinyHeader: 文件头，固定 192 字节，放在文件最前面
    //
    // 注意字段顺序是精心安排的: u64 字段都落在 8 的倍数偏移上，
    // 这样编译器不会插入 padding，内存布局 == 磁盘布局。
    // 这意味着可以直接把文件内容 memcpy 到结构体上，不需要逐字段解析。
    // -------------------------------------------------------------------------
    struct TinyHeader {
        char magic[8];                // 魔数，必须等于 kMagic ("TINYQWEN")
        uint32_t version;             // 格式版本，必须等于 kFormatVersion
        uint32_t dtype;               // 所有 tensor 的默认 dtype（0 = f32, 1 = f16, 3 = i4）
        // --- 模型配置（描述这是一个什么形状的模型）---
        uint32_t n_layers;            // transformer 层数
        uint32_t hidden_size;         // 隐藏层宽度（每个 token 的向量维度）
        uint32_t intermediate_size;   // FFN 中间层宽度
        uint32_t n_heads;             // query 注意力头数
        uint32_t n_kv_heads;          // key/value 头数（GQA，通常 <= n_heads）
        uint32_t head_dim;            // 每个注意力头的维度
        uint32_t vocab_size;          // 词表大小
        uint32_t max_seq_len;         // 训练时支持的最大序列长度
        uint32_t tied_embeddings;     // 1 = lm_head 与 embed_tokens 共享同一份权重
        uint32_t reserved_u32;        // 预留字段，必须填 0
        float rms_norm_eps;           // RMSNorm 里防止除零的小常数
        float rope_theta;             // RoPE 位置编码的底数 theta
        // --- 表区 / 数据区的位置 ---
        uint64_t tensor_count;        // tensor 表里有多少项
        uint64_t tensor_table_offset; // tensor 表的起始偏移（v1 固定 = 192）
        uint64_t data_offset;         // 第一个 tensor 数据的偏移，必须 % 64 == 0
        uint64_t total_bytes;         // 整个文件的字节数（加载时用来校验文件完整性）
        char reserved[96];            // v1 必须全 0；v2 按 TinyHeaderV2Ext 解读
    };

    // -------------------------------------------------------------------------
    // TinyHeaderV2Ext: v2 扩展头
    //
    // 复用 TinyHeader::reserved[96] 的前 64 字节。
    // version >= 2 时加载端把 reserved 按此结构 memcpy 出来读。
    // 导出端写 v2 文件时填这些字段，剩余字节补 0。
    // -------------------------------------------------------------------------
    struct TinyHeaderV2Ext {
        uint32_t model_type;                // ModelType: 0 = Qwen2.x, 1 = Qwen3.5 混合
        uint32_t linear_num_qk_heads;       // GDN 线性注意力的 Q/K 头数
        uint32_t linear_num_v_heads;        // GDN 线性注意力的 V 头数
        uint32_t linear_qk_head_dim;        // GDN 每个 Q/K 头的维度
        uint32_t linear_v_head_dim;         // GDN 每个 V 头的维度
        uint32_t linear_conv_kernel_dim;    // GDN 内 causal conv1d 的 kernel 大小
        uint32_t full_attention_interval;   // 每 N 层出现一个 full attention（3:1 -> 4）
        float partial_rotary_factor;        // RoPE 只旋转 head_dim 的该比例（如 0.25）
        uint32_t eos_token_id;              // stop token（Qwen3.5 = 248044）
        uint32_t pad;                       // 必须填 0，保持 8 字节对齐
        // --- 量化字段（仅 dtype == kI4 时有意义；非量化文件此处全 0）---
        uint32_t quant_group_size;          // INT4 每组元素数（128 典型）；0 = 未量化
        char ext_reserved[20];              // 余量，必须全 0（留给未来扩展）
    };
    // 编译期断言: 确保 V2 扩展头恰好 64 字节
    static_assert(sizeof(TinyHeaderV2Ext) == 64, "TinyHeaderV2Ext must be 64 bytes");

    // tensor 名字最长 64 字节（包括结尾的 NUL 字符）
    inline constexpr size_t kMaxTensorName = 64;

    // -------------------------------------------------------------------------
    // TensorEntry: tensor 表的一项
    //
    // 描述"某一个 tensor 在哪、多大、什么形状"。
    // 注意: 这里不存数据本身，只存"指向数据区的元信息"。
    // 数据本身存在文件的数据区，由 offset 和 nbytes 定位。
    // -------------------------------------------------------------------------
    struct TensorEntry {
        char name[kMaxTensorName];  // 名字，用 NUL 补齐；占满 64 字符时无终止符
        uint32_t dtype;             // 数据类型，见 Dtype 枚举
        uint32_t ndim;              // 维度数 1..4，shape 里后面对应的位才有效
        uint64_t shape[4];          // 形状；shape[ndim..3] 必须为 0
        uint64_t offset;            // 数据在文件里的绝对偏移，必须 % 64 == 0
        uint64_t nbytes;            // 数据字节数 = 元素个数 * 每元素字节数
    };

    // =========================================================================
    // 编译期断言: 把结构体大小"锁死"
    //
    // 一旦有人加字段导致编译器插入 padding、大小变了，编译立刻报错，
    // 避免"内存布局和磁盘布局悄悄不一致"这种极难排查的 bug。
    // =========================================================================
    static_assert(sizeof(TinyHeader) == 192, "TinyHeader must be exactly 192 bytes");
    static_assert(sizeof(TensorEntry) == 120, "TensorEntry must be exactly 120 bytes");
    static_assert(offsetof(TinyHeader, tensor_count) % 8 == 0,
                  "u64 fields must stay 8-byte aligned");
    static_assert(offsetof(TensorEntry, shape) % 8 == 0,
                  "u64 fields must stay 8-byte aligned");
} // namespace tinyqwen