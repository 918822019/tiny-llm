// GDN 状态容器实现。内存策略与 KvCache 相同：init 时一次分配整块 arena，
// 之后零分配；reset 只清零，不释放。

#include "gdn_state.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {
    void GdnState::init(int n_linear_layers, int num_v_heads, int qk_head_dim, int v_head_dim,
                        int conv_dim, int conv_kernel_size) {
        if (n_linear_layers <= 0 || num_v_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0 ||
            conv_dim <= 0 || conv_kernel_size < 2) {
            std::fprintf(stderr, "GdnState::init: bad dimensions (%d, %d, %d, %d, %d, %d)\n",
                         n_linear_layers, num_v_heads, qk_head_dim, v_head_dim, conv_dim,
                         conv_kernel_size);
            std::abort();
        }
        n_linear_layers_ = n_linear_layers;
        num_v_heads_ = num_v_heads;
        qk_head_dim_ = qk_head_dim;
        v_head_dim_ = v_head_dim;
        conv_dim_ = conv_dim;
        conv_state_len_ = conv_kernel_size - 1;

        rec_stride_ = static_cast<size_t>(num_v_heads) * qk_head_dim * v_head_dim;
        conv_stride_ = static_cast<size_t>(conv_dim) * conv_state_len_;
        conv_base_ = static_cast<size_t>(n_linear_layers) * rec_stride_;
        data_.assign(conv_base_ + static_cast<size_t>(n_linear_layers) * conv_stride_, 0.0f);
    }

    void GdnState::reset() {
        for (float &x : data_) x = 0.0f;
    }

    float *GdnState::recurrent(int linear_layer) {
        return data_.data() + static_cast<size_t>(linear_layer) * rec_stride_;
    }

    float *GdnState::conv(int linear_layer) {
        return data_.data() + conv_base_ + static_cast<size_t>(linear_layer) * conv_stride_;
    }
} // namespace tinyqwen
