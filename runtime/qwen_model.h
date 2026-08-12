#pragma once

// QwenModel：把权重、KV cache、workspace 组装成一个能跑的模型，
// 并提供 forward_token() 做"输入一个 token、输出下一个 token"。
//
// 数学定义对照 docs/qwen_forward.md；整体流程概念见 docs/infra_primer.md。

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache.h"
#include "model_loader.h"
#include "profiler.h"

namespace tinyqwen {

// top-k 结果：分数最高的 k 个 token 的 id 和对应 logits。
struct TopKResult {
  std::vector<int> indices;
  std::vector<float> values;
};

// 固定结构的 Qwen-like decoder-only 模型。
// batch = 1，token-by-token，greedy，fp32 reference 路径。
//
// 刻意不做"图/graph"抽象：forward 就是一个写死的函数，op 顺序和
// docs/qwen_forward.md 一一对应，这样跟 PyTorch 对不上时能肉眼定位是哪步错了。
// 通用图框架灵活但难调试，对"只跑一个模型"的我们来说是过度设计。
class QwenModel {
 public:
  // 工厂函数：校验所有需要的 tensor 都在、形状对，然后初始化 KV cache 和 buffer。
  // max_seq_len：运行时 KV 容量上限，必须 <= 文件头里的 max_seq_len。
  static bool create(const ModelFile& file, int max_seq_len, Profiler& profiler,
                     std::string* err, std::unique_ptr<QwenModel>* out);

  // 在当前位置（== kv_cache().seq_len()）上前向一个 token。
  // 返回 greedy 的下一个 token id；可选同时填 top-k logits。
  int forward_token(int token_id, TopKResult* topk = nullptr, int topk_k = 5);

  void reset();  // 清空 KV cache 和 token 计数（开始新的一段对话）
  void set_prompt_len(int n) { prompt_len_ = n; }  // 让 profiler 知道哪些是 prefill

  const ModelConfig& config() const { return cfg_; }
  KvCache& kv_cache() { return kv_; }
  int token_count() const { return token_count_; }
  // 最近一次 forward_token 的 logits（vocab_size 个 float，对词表每个词一个分数）。
  const float* last_logits() const { return logits_.data(); }

 private:
  QwenModel() = default;

  // 某一层的权重指针集合。都是裸指针，指向 ModelFile 内部的内存（只读、不拥有）。
  struct LayerWeights {
    const float* input_ln = nullptr;   // attention 前的 RMSNorm 权重
    const float* q_proj = nullptr;     // query 投影矩阵
    const float* k_proj = nullptr;     // key 投影矩阵
    const float* v_proj = nullptr;     // value 投影矩阵
    // Qwen2/2.5 的 attention 带 q/k/v bias（attention_bias=True）；o 无 bias。
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    const float* o_proj = nullptr;     // 输出投影矩阵
    const float* post_ln = nullptr;    // FFN 前的 RMSNorm 权重
    const float* gate = nullptr;       // SwiGLU 的 gate 投影
    const float* up = nullptr;         // SwiGLU 的 up 投影
    const float* down = nullptr;       // SwiGLU 的 down 投影
  };

  // 按名字取 tensor 并校验 shape；失败时填 *err 并返回 nullptr。
  const float* require(const ModelFile& file, const std::string& name,
                       const std::vector<uint64_t>& shape, std::string* err);

  ModelConfig cfg_{};
  Profiler* profiler_ = nullptr;
  int prompt_len_ = 0;
  int token_count_ = 0;
  int q_dim_ = 0;   // query 总维度 = n_heads * head_dim
  int kv_dim_ = 0;  // key/value 总维度 = n_kv_heads * head_dim
  int max_seq_len_ = 0;

  const float* embed_ = nullptr;      // 词嵌入表 [vocab, hidden]
  const float* final_norm_ = nullptr; // 最后的 RMSNorm
  const float* lm_head_ = nullptr;    // 输出投影到词表；tied 时 == embed_
  std::vector<LayerWeights> layers_;
  KvCache kv_;

  // workspace buffer：forward 时反复使用的临时空间，create() 时一次分配，
  // 避免每个 token 都 new/delete（那会很慢且产生碎片）。
  std::vector<float> hidden_, normed_, q_, k_, v_, attn_, o_, gate_, up_, ffn_, logits_;
};

}  // namespace tinyqwen
