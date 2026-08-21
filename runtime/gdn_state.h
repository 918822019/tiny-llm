#pragma once

// ============================================================================
// 文件: gdn_state.h
// 作用: GDN（Gated DeltaNet）线性注意力层的推理状态管理
//
// GDN 状态 vs KV Cache:
//   与 full attention 的 KV cache 不同，GDN 层不随序列长度增长:
//     - 递归状态 S: 把全部历史压缩进一个固定大小的矩阵（"记忆矩阵"），
//       每个 token 用 delta rule 原地更新；
//     - conv 状态: GDN 内部 causal conv1d 的滑动窗口，只保留最近
//       (kernel_size - 1) 个 token 的混合 qkv 投影值。
//
// 内存优势:
//   18 层 GDN 的总状态是 O(1) 的（与 max_seq_len 无关），这是 Qwen3.5
//   混合架构长上下文内存优势的核心。相比之下，full attention 的 KV cache
//   随序列长度线性增长。
//
// 布局对齐:
//   与 HuggingFace transformers 的 cache 对齐（便于逐位对齐验证）:
//     recurrent: [n_linear_layers][num_v_heads][qk_head_dim][v_head_dim]
//     conv:      [n_linear_layers][conv_dim][conv_kernel_size - 1]
//
// 前置阅读: docs/infra_primer.md（内存布局与指针）
// ============================================================================

#include <cstddef>
#include <vector>

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // GdnState: GDN 线性注意力层的推理状态容器
    //
    // 管理两种状态:
    //   1. 递归状态 (recurrent): 每个 GDN 层的每个 v head 一个
    //      [qk_head_dim, v_head_dim] 的矩阵，用 delta rule 更新
    //   2. 卷积状态 (conv): 每个 GDN 层的 causal conv1d 滑动窗口
    //      保存最近 (kernel_size - 1) 个 token 的混合 qkv 投影值
    //
    // 内存布局:
    //   一整块 arena: 先全部 recurrent 状态，再全部 conv 状态
    //     recurrent 区: [n_linear_layers][num_v_heads][qk_head_dim][v_head_dim]
    //     conv 区:      [n_linear_layers][conv_dim][conv_kernel_size - 1]
    //
    // 使用场景:
    //   - QwenModel 在 create() 时初始化 GdnState
    //   - forward 中每遇到 GDN 层，从 GdnState 取对应层的 recurrent/conv 状态
    //   - 调用 gdn_step 等 kernel 就地更新状态
    //   - reset() 清空全部状态，开始新对话
    // -------------------------------------------------------------------------
    class GdnState {
    public:
        GdnState() = default;

        // ---------------------------------------------------------------------
        // init: 分配 GDN 状态所需的内存
        //
        // 参数:
        //   n_linear_layers:  GDN 层的数量（qwen3_5: n_layers - n_full_layers）
        //   num_v_heads:      每个 GDN 层的 V 注意力头数
        //   qk_head_dim:      每个 Q/K 头的维度
        //   v_head_dim:       每个 V 头的维度
        //   conv_dim:         卷积通道数（= 2 * qk_head_dim * num_qk_heads + v_head_dim * num_v_heads）
        //   conv_kernel_size: causal conv1d 的 kernel 大小（Qwen3.5 = 4）
        //
        // 说明:
        //   采用 arena 分配: 一次性分配所有层、所有状态的内存。
        //   优点: 内存连续、cache 友好、避免碎片。
        // ---------------------------------------------------------------------
        void init(int n_linear_layers, int num_v_heads, int qk_head_dim, int v_head_dim,
                  int conv_dim, int conv_kernel_size);

        // ---------------------------------------------------------------------
        // reset: 全部清零
        //
        // 说明:
        //   将 recurrent 和 conv 状态全部置零，开始新的一段对话。
        //   保留已分配的内存，不释放。
        // ---------------------------------------------------------------------
        void reset();

        // 是否已初始化（分配了内存）
        bool initialized() const { return !data_.empty(); }

        // ---------------------------------------------------------------------
        // recurrent: 获取第 linear_layer 个 GDN 层的递归状态
        //
        // 参数:
        //   linear_layer: GDN 层的紧凑索引（0-based，不是全局层号）
        //
        // 返回值:
        //   指向递归状态矩阵的 float* 指针，布局为 [num_v_heads][qk_head_dim][v_head_dim]
        //
        // 说明:
        //   调用方拿到指针后，按以下偏移计算具体位置:
        //     某个 v head h 的状态矩阵起点 = h * (qk_head_dim * v_head_dim)
        //     矩阵元素 (i, j) = i * v_head_dim + j（行主序）
        // ---------------------------------------------------------------------
        float *recurrent(int linear_layer);

        // ---------------------------------------------------------------------
        // conv: 获取第 linear_layer 个 GDN 层的卷积状态
        //
        // 参数:
        //   linear_layer: GDN 层的紧凑索引（0-based）
        //
        // 返回值:
        //   指向卷积状态的 float* 指针，布局为 [conv_dim][conv_state_len]
        //   其中 conv_state_len = conv_kernel_size - 1
        //
        // 说明:
        //   卷积状态保存每个通道最近 (kernel_size - 1) 个历史输入，
        //   在 causal_conv1d_update 中作为滑动窗口使用。
        // ---------------------------------------------------------------------
        float *conv(int linear_layer);

        // GDN 层的总数量
        int n_linear_layers() const { return n_linear_layers_; }

        // ---------------------------------------------------------------------
        // memory_bytes: 状态占用的总字节数
        //
        // 返回值:
        //   整个 GDN 状态占用的内存字节数 = data_.size() * sizeof(float)
        //
        // 说明:
        //   用于打印内存占用信息。Qwen2.x 的 GDN 状态恒为 0（无 GDN 层）。
        // ---------------------------------------------------------------------
        size_t memory_bytes() const { return data_.size() * sizeof(float); }

    private:
        int n_linear_layers_ = 0;  // GDN 层的数量
        int num_v_heads_ = 0;      // 每个 GDN 层的 V 头数
        int qk_head_dim_ = 0;      // Q/K 头的维度
        int v_head_dim_ = 0;       // V 头的维度
        int conv_dim_ = 0;         // 卷积通道数
        int conv_state_len_ = 0;   // 卷积状态长度 = conv_kernel_size - 1

        // 一层递归状态占多少个 float = num_v_heads * qk_head_dim * v_head_dim
        size_t rec_stride_ = 0;

        // 一层卷积状态占多少个 float = conv_dim * conv_state_len
        size_t conv_stride_ = 0;

        // conv 区在 data_ 里的起点（float 元素下标）
        // recurrent 区从 0 开始，conv 区从 rec_stride_ * n_linear_layers 开始
        size_t conv_base_ = 0;

        // 一整块 arena 内存
        // 总大小 = n_linear_layers * (rec_stride_ + conv_stride_) 个 float
        std::vector<float> data_;
    };
} // namespace tinyqwen