#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "model_loader.h"
#include "profiler.h"

namespace tinyqwen {

struct TopKResult {
  std::vector<int> indices;
  std::vector<float> values;
};

// 固定结构的 Qwen-like decoder-only 模型。
// batch = 1，token-by-token，greedy，fp32 reference 路径。
//
// 不做图抽象：forward 是一个写死的函数，用显式 buffer，
// 计算顺序与 docs/qwen_forward.md 一一对应，方便人工审查。
class QwenModel {
 public:
  // 校验所有必需 tensor（name + shape）与 header 配置一致。
  // max_seq_len：运行时 KV 容量，必须 <= header 的 max_seq_len。
  static bool create(const ModelFile& file, int max_seq_len, Profiler& profiler,
                     std::string* err, std::unique_ptr<QwenModel>* out);

  // 在当前位置（== kv_cache().seq_len()）上跑一个 token。
  // 返回 greedy 的下一个 token id；可选填 top-k logits（topk_k）。
  int forward_token(int token_id, TopKResult* topk = nullptr, int topk_k = 5);

  void reset();  // 清空 KV cache 和 token 计数
  void set_prompt_len(int n) { prompt_len_ = n; }  // 供 profiler 判断 is_prefill

  const ModelConfig& config() const { return cfg_; }
  KvCache& kv_cache() { return kv_; }
  int token_count() const { return token_count_; }
  // 最近一次 forward_token 的 logits（vocab_size 个 float）。
  const float* last_logits() const { return logits_.data(); }

 private:
  QwenModel() = default;

  // 每层权重的裸指针视图（全部指向 ModelFile 内部，不拥有内存）。
  struct LayerWeights {
    const float* input_ln = nullptr;
    const float* q_proj = nullptr;
    const float* k_proj = nullptr;
    const float* v_proj = nullptr;
    // Qwen2/2.5 attention 带 q/k/v bias（attention_bias=True）；o 无 bias。
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    const float* o_proj = nullptr;
    const float* post_ln = nullptr;
    const float* gate = nullptr;
    const float* up = nullptr;
    const float* down = nullptr;
  };

  // 按名字取 tensor 并校验 shape，失败时填 *err 并返回 nullptr。
  const float* require(const ModelFile& file, const std::string& name,
                       const std::vector<uint64_t>& shape, std::string* err);

  ModelConfig cfg_{};
  Profiler* profiler_ = nullptr;
  int prompt_len_ = 0;
  int token_count_ = 0;
  int q_dim_ = 0;   // n_heads * head_dim
  int kv_dim_ = 0;  // n_kv_heads * head_dim
  int max_seq_len_ = 0;

  const float* embed_ = nullptr;
  const float* final_norm_ = nullptr;
  const float* lm_head_ = nullptr;  // tied 时 == embed_
  std::vector<LayerWeights> layers_;
  KvCache kv_;

  // workspace buffer，create() 时一次分配，forward 中不再分配。
  std::vector<float> hidden_, normed_, q_, k_, v_, attn_, o_, gate_, up_, ffn_, logits_;
};

}  // namespace tinyqwen
