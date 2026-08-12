#pragma once

#include <cstddef>
#include <cstdint>

namespace tinyqwen {

// ---------------------------------------------------------------------------
// tinyqwen flat binary format (".tqwen"), version 1.
//
// File layout (all integers little-endian, all offsets absolute from file
// start, exactly as written on disk — no struct padding allowed to leak):
//
//   [ TinyHeader                     ]  192 bytes
//   [ TensorEntry * tensor_count     ]  120 bytes each, starts at offset 192
//   [ padding to 64-byte alignment   ]
//   [ tensor payloads                ]  each starts at a 64-byte aligned offset
//
// This header is the single source of truth shared by:
//   - C++ loader   (runtime/model_loader.cpp)
//   - Python exporter (tools/export_qwen_to_tiny.py)
// Any change here requires bumping kFormatVersion AND updating the exporter.
// See docs/weight_format.md for the full specification.
// ---------------------------------------------------------------------------

inline constexpr char kMagic[8] = {'T', 'I', 'N', 'Y', 'Q', 'W', 'E', 'N'};
inline constexpr uint32_t kFormatVersion = 1;
inline constexpr size_t kAlignment = 64;

enum class Dtype : uint32_t {
  kF32 = 0,  // v1 only supported payload dtype
  kF16 = 1,  // reserved
  kI8 = 2,   // reserved: weight-only INT8 (per-channel / per-group scale)
  kI4 = 3,   // reserved: packed INT4 / KronQ (sub-byte, special layout)
};

// Bytes per element; returns 0 for unknown / sub-byte dtypes.
size_t dtype_size(Dtype dtype);
const char* dtype_name(Dtype dtype);

struct TinyHeader {
  char magic[8];            // kMagic
  uint32_t version;         // kFormatVersion
  uint32_t dtype;           // default dtype of all tensors (v1: 0 = f32)
  // --- model config (Qwen-like decoder-only) ---
  uint32_t n_layers;
  uint32_t hidden_size;
  uint32_t intermediate_size;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t vocab_size;
  uint32_t max_seq_len;     // from HF config max_position_embeddings
  uint32_t tied_embeddings;   // 1 => lm_head shares model.embed_tokens.weight
  uint32_t reserved_u32;    // must be 0
  float rms_norm_eps;
  float rope_theta;
  // --- table / payload location ---
  uint64_t tensor_count;
  uint64_t tensor_table_offset;  // == sizeof(TinyHeader) in v1
  uint64_t data_offset;          // first tensor payload, % kAlignment == 0
  uint64_t total_bytes;          // full file size in bytes
  char reserved[96];             // must be 0
};

inline constexpr size_t kMaxTensorName = 64;

struct TensorEntry {
  char name[kMaxTensorName];  // NUL-padded; may not be terminated when full
  uint32_t dtype;             // Dtype
  uint32_t ndim;              // 1..4, shape[ndim..] == 0
  uint64_t shape[4];
  uint64_t offset;            // absolute byte offset of payload, % 64 == 0
  uint64_t nbytes;            // numel * dtype_size(dtype)
};

static_assert(sizeof(TinyHeader) == 192, "TinyHeader must be exactly 192 bytes");
static_assert(sizeof(TensorEntry) == 120, "TensorEntry must be exactly 120 bytes");
static_assert(offsetof(TinyHeader, tensor_count) % 8 == 0,
              "u64 fields must stay 8-byte aligned");
static_assert(offsetof(TensorEntry, shape) % 8 == 0,
              "u64 fields must stay 8-byte aligned");

}  // namespace tinyqwen
