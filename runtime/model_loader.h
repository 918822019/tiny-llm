#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "tiny_format.h"

namespace tinyqwen {

// 从文件 header 解析出的模型配置。
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

// 加载并校验 .tqwen 文件。拥有文件字节；TensorView 的 data 指针
// 指向本对象内部。不可拷贝。
class ModelFile {
 public:
  ModelFile() = default;
  ModelFile(const ModelFile&) = delete;
  ModelFile& operator=(const ModelFile&) = delete;

  // 读入整个文件并校验 header / tensor 表 / 数据区。
  // 任何一步失败都返回 false 并通过 *err（非空时）给出原因。
  bool load(const std::string& path, std::string* err);

  bool loaded() const { return !data_.empty(); }
  const uint8_t* base() const { return data_.data(); }
  const TinyHeader& header() const { return header_; }
  const ModelConfig& config() const { return config_; }

  // tensor 不存在时返回 nullptr。
  const TensorView* get(const std::string& name) const;
  size_t tensor_count() const { return order_.size(); }
  const std::vector<std::string>& tensor_names() const { return order_; }

  void print_summary() const;

 private:
  std::vector<uint8_t> data_;
  TinyHeader header_{};
  ModelConfig config_{};
  std::unordered_map<std::string, TensorView> tensors_;
  std::vector<std::string> order_;  // 按文件内顺序记录名字，供 summary 打印
};

}  // namespace tinyqwen
