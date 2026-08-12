#pragma once

#include <cstddef>
#include <cstdint>

namespace tinyqwen {

// ---------------------------------------------------------------------------
// tinyqwen 扁平二进制格式（".tqwen"），版本 1。
//
// 文件布局（所有整数小端，所有 offset 均为文件内绝对偏移，
// 磁盘内容与结构体逐字节一致——不允许结构体 padding 混入）：
//
//   [ TinyHeader                     ]  192 字节
//   [ TensorEntry * tensor_count     ]  每个 120 字节，从偏移 192 开始
//   [ 64 字节对齐的 padding          ]
//   [ tensor 数据区                  ]  每个 tensor 起始偏移均 64 字节对齐
//
// 本头文件是 C++ 与 Python 共享的唯一契约，同时约束：
//   - C++ loader（runtime/model_loader.cpp）
//   - Python exporter（tools/export_qwen_to_tiny.py）
// 任何修改都必须 bump kFormatVersion 并同步修改 exporter。
// 完整规范见 docs/weight_format.md。
// ---------------------------------------------------------------------------

inline constexpr char kMagic[8] = {'T', 'I', 'N', 'Y', 'Q', 'W', 'E', 'N'};
inline constexpr uint32_t kFormatVersion = 1;
inline constexpr size_t kAlignment = 64;

enum class Dtype : uint32_t {
  kF32 = 0,  // v1 唯一支持的数据类型
  kF16 = 1,  // 保留
  kI8 = 2,   // 保留：weight-only INT8（per-channel / per-group scale）
  kI4 = 3,   // 保留：打包 INT4 / KronQ（亚字节，专用布局）
};

// 每个元素的字节数；未知/亚字节类型返回 0。
size_t dtype_size(Dtype dtype);
const char* dtype_name(Dtype dtype);

struct TinyHeader {
  char magic[8];            // kMagic
  uint32_t version;         // kFormatVersion
  uint32_t dtype;           // 全部 tensor 的默认 dtype（v1: 0 = f32）
  // --- 模型配置（Qwen-like decoder-only）---
  uint32_t n_layers;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t vocab_size;
  uint32_t max_seq_len;     // 来自 HF config 的 max_position_embeddings
  uint32_t tied_embeddings;   // 1 => lm_head 与 model.embed_tokens.weight 共享
  uint32_t reserved_u32;    // 必须为 0
  float rms_norm_eps;
  float rope_theta;
  // --- 表区 / 数据区位置 ---
  uint64_t tensor_count;
  uint64_t tensor_table_offset;  // v1 中 == sizeof(TinyHeader)
  uint64_t data_offset;          // 第一个 tensor 数据偏移，% kAlignment == 0
  uint64_t total_bytes;          // 文件总字节数
  char reserved[96];             // 必须为 0
};

inline constexpr size_t kMaxTensorName = 64;

struct TensorEntry {
  char name[kMaxTensorName];  // NUL 补齐；占满 64 字符时无终止符
  uint32_t dtype;             // Dtype
  uint32_t ndim;              // 1..4，shape[ndim..] == 0
  uint64_t shape[4];
  uint64_t offset;            // 数据绝对偏移，% 64 == 0
  uint64_t nbytes;            // numel * dtype_size(dtype)
};

static_assert(sizeof(TinyHeader) == 192, "TinyHeader must be exactly 192 bytes");
static_assert(sizeof(TensorEntry) == 120, "TensorEntry must be exactly 120 bytes");
static_assert(offsetof(TinyHeader, tensor_count) % 8 == 0,
              "u64 fields must stay 8-byte aligned");
static_assert(offsetof(TensorEntry, shape) % 8 == 0,
              "u64 fields must stay 8-byte aligned");

}  // namespace tinyqwen
