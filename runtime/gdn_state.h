#pragma once

// GDN（Gated DeltaNet）线性注意力层的推理状态。
//
// 与 full attention 的 KV cache 不同，GDN 层不随序列长度增长：
//   - 递归状态 S：把全部历史压缩进一个固定大小的矩阵（"记忆矩阵"），
//     每个 token 用 delta rule 原地更新；
//   - conv 状态：GDN 内部 causal conv1d 的滑动窗口（最近 kernel-1 个
//     token 的混合 qkv 投影值）。
//
// 因此 18 层 GDN 的总状态是 O(1) 的（与 max_seq_len 无关），这是
// Qwen3.5 混合架构长上下文内存优势的核心。
//
// 布局与 HuggingFace transformers 的 cache 对齐（便于逐位对齐验证）：
//   recurrent: [n_linear_layers][num_v_heads][qk_head_dim][v_head_dim]
//   conv     : [n_linear_layers][conv_dim][conv_kernel_size - 1]

#include <cstddef>
#include <vector>

namespace tinyqwen {
    class GdnState {
    public:
        GdnState() = default;

        // n_linear_layers：GDN 层的数量（qwen3_5：n_layers - n_full_layers）。
        // conv_kernel_size：causal conv1d 的 kernel 大小（Qwen3.5 = 4）。
        void init(int n_linear_layers, int num_v_heads, int qk_head_dim, int v_head_dim,
                  int conv_dim, int conv_kernel_size);

        void reset(); // 全部清零（开始新的一段对话）

        bool initialized() const { return !data_.empty(); }

        // 第 linear_layer 个 GDN 层的递归状态：[num_v_heads][qk_head_dim][v_head_dim]。
        float *recurrent(int linear_layer);

        // 第 linear_layer 个 GDN 层的 conv 状态：[conv_dim][conv_kernel_size - 1]。
        float *conv(int linear_layer);

        int n_linear_layers() const { return n_linear_layers_; }

        size_t memory_bytes() const { return data_.size() * sizeof(float); }

    private:
        int n_linear_layers_ = 0;
        int num_v_heads_ = 0;
        int qk_head_dim_ = 0;
        int v_head_dim_ = 0;
        int conv_dim_ = 0;
        int conv_state_len_ = 0; // = conv_kernel_size - 1
        size_t rec_stride_ = 0; // 一层递归状态占多少个 float
        size_t conv_stride_ = 0; // 一层 conv 状态占多少个 float
        size_t conv_base_ = 0; // conv 区在 data_ 里的起点（元素下标）
        std::vector<float> data_; // 一整块 arena：先全部 recurrent，再全部 conv
    };
} // namespace tinyqwen
