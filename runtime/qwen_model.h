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
        static bool create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out);

        // 在当前位置（== kv_cache().seq_len()）上前向一个 token。
        // 返回 greedy 的下一个 token id；可选同时填 top-k logits。
        int forward_token(int token_id, TopKResult *topk = nullptr, int topk_k = 5);

        void reset(); // 清空 KV cache 和 token 计数（开始新的一段对话）
        void set_prompt_len(int n) { prompt_len_ = n; } // 让 profiler 知道哪些是 prefill

        const ModelConfig &config() const { return cfg_; }
        Dtype dtype() const { return dtype_; } // 权重精度（f32 / f16）
        KvCache &kv_cache() { return kv_; }
        int token_count() const { return token_count_; }
        // 最近一次 forward_token 的 logits（vocab_size 个 float，对词表每个词一个分数）。
        const float *last_logits() const { return logits_.data(); }

    private:
        QwenModel() = default;

        // 某一层的权重指针集合。都是裸指针（只读、不拥有），但分两类：
        //   - 大矩阵（proj/FFN）：const void*，dtype 跟随模型文件（f32/f16），
        //     forward 按 dtype_ 走 matvec_f32 / matvec_f16；
        //   - 小向量（norm/bias）：恒为 fp32 指针——f32 模型直接指向文件内存；
        //     f16 模型在 create() 时转成 fp32 副本存进 owned_f32_（量级 KB，
        //     换来 rmsnorm/bias 热点路径完全不改）。
        struct LayerWeights {
            const float *input_ln = nullptr; // attention 前的 RMSNorm 权重
            const void *q_proj = nullptr; // query 投影矩阵
            const void *k_proj = nullptr; // key 投影矩阵
            const void *v_proj = nullptr; // value 投影矩阵
            // Qwen2/2.5 的 attention 带 q/k/v bias（attention_bias=True）；o 无 bias。
            const float *q_bias = nullptr;
            const float *k_bias = nullptr;
            const float *v_bias = nullptr;
            const void *o_proj = nullptr; // 输出投影矩阵
            const float *post_ln = nullptr; // FFN 前的 RMSNorm 权重
            const void *gate = nullptr; // SwiGLU 的 gate 投影
            const void *up = nullptr; // SwiGLU 的 up 投影
            const void *down = nullptr; // SwiGLU 的 down 投影
        };

        // 按名字取 tensor 并校验 ndim/shape/dtype（== 文件 dtype）；
        // 失败时填 *err 并返回 nullptr。
        const TensorView *require_view(const ModelFile &file, const std::string &name,
                                       const std::vector<uint64_t> &shape, std::string *err);

        // 小向量绑定为 fp32：f32 模型直接返回文件内指针；f16 模型转换进
        // owned_f32_ 后返回副本指针（见 LayerWeights 注释）。
        const float *bind_f32_vector(const TensorView *t);

        // matvec 分派薄封装：按 dtype_ 调 matvec_f32 / matvec_f16 通用入口。
        void mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const;
        void mv_pair(const void *w1, const void *w2, const float *x, float *y1, float *y2,
                     int out_dim, int in_dim) const;

        ModelConfig cfg_{};
        Profiler *profiler_ = nullptr;
        int prompt_len_ = 0;
        int token_count_ = 0;
        int q_dim_ = 0; // query 总维度 = n_heads * head_dim
        int kv_dim_ = 0; // key/value 总维度 = n_kv_heads * head_dim
        int max_seq_len_ = 0;

        Dtype dtype_ = Dtype::kF32; // 权重 dtype（= 文件 header dtype），全模型单一

        const void *embed_ = nullptr; // 词嵌入表 [vocab, hidden]（dtype 随模型）
        const float *final_norm_ = nullptr; // 最后的 RMSNorm（恒 fp32，见 bind_f32_vector）
        const void *lm_head_ = nullptr; // 输出投影到词表；tied 时 == embed_
        // f16 模型专用：norm/bias 小向量的 fp32 副本（create 时一次转换）。
        // 注意：外层 vector 扩容只会移动 vector 对象本身，其堆内存（data()
        // 指向处）不动，所以已返回的 float* 不会悬空。
        std::vector<std::vector<float>> owned_f32_;
        std::vector<LayerWeights> layers_;
        KvCache kv_;

        // workspace buffer：forward 时反复使用的临时空间，create() 时一次分配，
        // 避免每个 token 都 new/delete（那会很慢且产生碎片）。
        std::vector<float> hidden_, normed_, q_, k_, v_, attn_, o_, gate_, up_, ffn_, logits_;
    };
} // namespace tinyqwen
