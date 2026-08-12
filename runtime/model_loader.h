#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "tiny_format.h"

namespace tinyqwen {

// Model config parsed from the file header.
struct ModelConfig {
  uint32_t n_layers = 0;
  uint32_t hidden_size = 0;
  uint32_t intermediate_size = 0;
  uint32_t n_heads = 0;
  uint32_t n_kv_heads = 0;
  uint32_t head_dim = 0;
  uint32_t vocab_size = 0;
  uint32_t max_seq_len = 0;
  float rms_norm_eps = 0.0f;
  float rope_theta = 0.0f;
  bool tied_embeddings = false;
};

// Loads and validates a .tqwen file. Owns the file bytes; TensorView data
// pointers point inside this object. Not copyable.
class ModelFile {
 public:
  ModelFile() = default;
  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;

  // Reads the whole file and validates header / table / payloads.
  // Returns false and fills *err (when non-null) on any failure.
  bool load(const std::string& path, std::string* err);

  bool loaded() const { return !data_.empty(); }
  const uint8_t* base() const { return data_.data(); }
  const TinyHeader& header() const { return header_; }
  const ModelConfig& config() const { return config_; }

  // Returns nullptr when the tensor does not exist.
  const TensorView* get(const std::string& name) const;
  size_t tensor_count() const { return order_.size(); }
  const std::vector<std::string>& tensor_names() const { return order_; }

  void print_summary() const;

 private:
  std::vector<uint8_t> data_;
  TinyHeader header_{};
  ModelConfig config_{};
  std::unordered_map<std::string, TensorView> tensors_;
  std::vector<std::string> order_;
};

}  // namespace tinyqwen
