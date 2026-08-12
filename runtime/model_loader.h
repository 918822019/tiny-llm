#pragma once

// ModelFile：把 .tqwen 文件读进内存并校验，然后提供按名字取 tensor 的能力。
// 先看 docs/infra_primer.md 第 5、9 节（二进制格式 / fail fast）。

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "tiny_format.h"

namespace tinyqwen {
    // 从文件头解析出来的模型配置（这是一个什么形状的模型）。
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

    // 加载并校验一个 .tqwen 文件。
    //
    // 内存所有权：本对象把整个文件读进自己内部的 data_ buffer；所有 TensorView
    // 的 data 指针都指向 data_ 内部。因此 ModelFile 必须比所有使用这些指针的
    // 地方活得更久（见 primer 第 3 节"视图"）。因为持有这块内存，所以禁止拷贝。
    class ModelFile {
    public:
        ModelFile() = default;

        ModelFile(const ModelFile &) = delete; // 禁止拷贝（避免误拷贝整块权重内存）
        ModelFile &operator=(const ModelFile &) = delete;

        // 读入整个文件并逐层校验。任何一步失败都返回 false 并把原因写进 *err。
        bool load(const std::string &path, std::string *err);

        bool loaded() const { return !data_.empty(); }
        const uint8_t *base() const { return data_.data(); } // 文件数据在内存里的起点
        const TinyHeader &header() const { return header_; }
        const ModelConfig &config() const { return config_; }

        // 按名字取 tensor；不存在返回 nullptr。
        const TensorView *get(const std::string &name) const;

        size_t tensor_count() const { return order_.size(); }
        const std::vector<std::string> &tensor_names() const { return order_; }

        // 打印模型摘要（配置 + 每个 tensor 的 shape/偏移），调试用。
        void print_summary() const;

    private:
        std::vector<uint8_t> data_; // 整个文件的字节（我们拥有的唯一一份内存）
        TinyHeader header_{};
        ModelConfig config_{};
        std::unordered_map<std::string, TensorView> tensors_; // 名字 -> 视图
        std::vector<std::string> order_; // 按文件内顺序记录名字，供 summary 打印
    };
} // namespace tinyqwen
