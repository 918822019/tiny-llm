// ============================================================================
// gdn_state.cpp — GDN（Gated DeltaNet）状态容器实现
// ============================================================================
// 本文件实现 GdnState 类，管理 Qwen3.5 混合架构中 GDN linear attention 层的
// 状态张量。GDN 是一种线性注意力机制，其核心优势是 O(1) 的推理复杂度（与
// 序列长度无关），内存占用也保持不变。
//
// GDN 每层需要维护两类状态：
//   1. 递归状态（recurrent state）：
//      形状 [num_v_heads, qk_head_dim, v_head_dim]
//      用于 delta rule 的矩阵递归更新，类似于线性 RNN 的隐藏状态。
//   2. 卷积状态（conv state）：
//      形状 [conv_dim, conv_kernel_size - 1]
//      用于 causal conv1d 的延迟线缓冲区，存储最近几次输入。
//
// 内存策略与 KvCache 相同：init 时一次分配整块 arena，之后零分配；
// reset 只清零，不释放。arena 布局：
//   [Layer 0 recurrent state][Layer 1 recurrent state]...[Layer N-1 recurrent state]
//   [Layer 0 conv state][Layer 1 conv state]...[Layer N-1 conv state]
// ============================================================================

#include "gdn_state.h"

#include <cstdio>   // 标准输入输出
#include <cstdlib>  // 标准库（abort）

namespace tinyqwen {
    // =========================================================================
    // GdnState::init() — 初始化 GDN 状态
    // =========================================================================
    // 参数：
    //   n_linear_layers  — GDN linear attention 层的数量
    //   num_v_heads      — value 头数量
    //   qk_head_dim      — query/key 头的维度
    //   v_head_dim       — value 头的维度
    //   conv_dim         — 卷积输入维度（2 * qk_dim + v_dim）
    //   conv_kernel_size — 卷积核大小（至少为 2）
    // 说明：计算 arena 总大小，一次性分配并清零。所有参数必须为正，conv_kernel_size
    //       至少为 2（否则卷积没有意义）。直接发声 abort 而非返回错误码，
    //       因为这是编程错误而非运行时错误。
    void GdnState::init(int n_linear_layers, int num_v_heads, int qk_head_dim, int v_head_dim,
                        int conv_dim, int conv_kernel_size) {
        // 参数合法性检查
        if (n_linear_layers <= 0 || num_v_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
            conv_dim <= 0 || conv_kernel_size < 2) {
            std::fprintf(stderr, "GdnState::init: bad dimensions (%d, %d, %d, %d, %d, %d)\n",
                         n_linear_layers, num_v_heads, qk_head_dim, v_head_dim, conv_dim,
                         conv_kernel_size);
            std::abort(); // 参数错误，立即终止
        }
        n_linear_layers_ = n_linear_layers;
        num_v_heads_ = num_v_heads;
        qk_head_dim_ = qk_head_dim;
        v_head_dim_ = v_head_dim;
        conv_dim_ = conv_dim;
        // 卷积状态存储最近 (kernel_size - 1) 个时间步的输入
        conv_state_len_ = conv_kernel_size - 1;

        // 每层递归状态的步长：num_v_heads * qk_head_dim * v_head_dim 个 float
        rec_stride_ = static_cast<size_t>(num_v_heads) * qk_head_dim * v_head_dim;
        // 每层卷积状态的步长：conv_dim * conv_state_len_ 个 float
        conv_stride_ = static_cast<size_t>(conv_dim) * conv_state_len_;
        // 卷积状态在 arena 中的起始偏移：在所有递归状态之后
        conv_base_ = static_cast<size_t>(n_linear_layers) * rec_stride_;
        // 一次性分配所有内存并清零
        data_.assign(conv_base_ + static_cast<size_t>(n_linear_layers) * conv_stride_, 0.0f);
    }

    // =========================================================================
    // GdnState::reset() — 重置所有状态为零
    // =========================================================================
    // 说明：将整个 arena 清零。调用时机：每条新 prompt 开始时。
    void GdnState::reset() {
        for (float &x : data_) x = 0.0f; // 逐元素清零
    }

    // =========================================================================
    // GdnState::recurrent() — 获取指定层的递归状态起始地址
    // =========================================================================
    // 参数：
    //   linear_layer — linear attention 层的本地索引（0 到 n_linear_layers_-1）
    // 返回值：该层递归状态第一个元素的指针
    // 说明：递归状态形状 [num_v_heads, qk_head_dim, v_head_dim]，按行优先存储。
    float *GdnState::recurrent(int linear_layer) {
        return data_.data() + static_cast<size_t>(linear_layer) * rec_stride_;
    }

    // =========================================================================
    // GdnState::conv() — 获取指定层的卷积状态起始地址
    // =========================================================================
    // 参数：
    //   linear_layer — linear attention 层的本地索引（0 到 n_linear_layers_-1）
    // 返回值：该层卷积状态第一个元素的指针
    // 说明：卷积状态形状 [conv_dim, conv_state_len_]，存储最近 conv_state_len_
    //       个时间步的输入，用于 causal conv1d 的延迟线。
    float *GdnState::conv(int linear_layer) {
        // 卷积状态在递归状态之后，需要加上 conv_base_ 偏移
        return data_.data() + conv_base_ + static_cast<size_t>(linear_layer) * conv_stride_;
    }
} // namespace tinyqwen