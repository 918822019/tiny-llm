#include "tensor.h"

#include <cstdio>
#include <cstdlib>

namespace tinyqwen {
    // 每种类型一个元素占几个字节。i4 是亚字节（一个字节装两个数），
    // 没法用"每元素字节数"描述，所以返回 0，由具体 kernel 特殊处理。
    size_t dtype_size(Dtype dtype) {
        switch (dtype) {
            case Dtype::kF32: return 4;
            case Dtype::kF16: return 2;
            case Dtype::kI8: return 1;
            case Dtype::kI4: return 0;
        }
        return 0;
    }

    const char *dtype_name(Dtype dtype) {
        switch (dtype) {
            case Dtype::kF32: return "f32";
            case Dtype::kF16: return "f16";
            case Dtype::kI8: return "i8";
            case Dtype::kI4: return "i4";
        }
        return "?";
    }

    uint64_t TensorView::numel() const {
        uint64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= shape[i];
        return n;
    }

    const float *TensorView::f32() const {
        if (dtype != Dtype::kF32) {
            std::fprintf(stderr, "tinyqwen: tensor '%s' has dtype %s, f32 expected\n",
                         name.c_str(), dtype_name(dtype));
            std::abort();
        }
        // data 是 uint8_t*（按字节寻址）；权重要按 float 读，所以 reinterpret 成
        // float*。这要求 data 地址对 float 对齐——我们的 64B 对齐保证了这一点。
        return reinterpret_cast<const float *>(data);
    }
} // namespace tinyqwen
