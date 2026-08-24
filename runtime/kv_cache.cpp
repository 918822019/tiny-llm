// ============================================================================
// kv_cache.cpp — KV Cache 的内存管理（含 fp16 量化存储）
// ============================================================================
// 在自回归推理中，每个 token 的 attention 计算需要访问所有之前 token 的
// Key 和 Value。为了避免每步都重新计算，我们将已计算过的 K 和 V 缓存起来。
//
// 设计要点:
//   1. 一次申请一整块连续内存（arena），而不是每层各申请一次。
//   2. arena 内部切成两大块：前半是 K，后半是 V。
//   3. 每块再按层切成 n_layers 个"平面"，每个平面是
//      [n_kv_heads][max_seq_len][head_dim] 的三维数组。
//   4. 追加新 token: write_token() 写入第 pos 个槽位，再 advance() 提交 +1。
//
// 内存布局（arena 整体）：
//   [Layer 0 K plane][Layer 1 K plane]...[Layer N-1 K plane]
//   [Layer 0 V plane][Layer 1 V plane]...[Layer N-1 V plane]
// 每个 plane 的大小 = n_kv_heads * max_seq_len * head_dim 个元素
//
// fp16 量化（v2）:
//   use_fp16=true 时 arena 用 uint16_t（fp16）存储，内存减半。
//   写入（write_token）: fp32 → fp16。
//   读取（k()/v()）: fp16 → fp32 反量化到 deq_ws_ workspace，attention 读
//     到的是 fp32，计算仍在 fp32 进行（存 fp16、算 fp32）。
// ============================================================================

#include "kv_cache.h"

#include <cstdio>   // fprintf
#include <cstdlib>  // abort
#include <cstring>  // memcpy

#include "ref_ops.h"  // half_to_float / float_to_half（fp16<->fp32 转换）

namespace tinyqwen {
    KvCache::KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim,
                     bool use_fp16) {
        init(n_layers, n_kv_heads, max_seq_len, head_dim, use_fp16);
    }

    void KvCache::init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim,
                       bool use_fp16) {
        if (n_layers <= 0 || n_kv_heads <= 0 || max_seq_len <= 0 || head_dim <= 0) {
            std::fprintf(stderr, "tinyqwen: KvCache::init with invalid dims\n");
            std::abort();
        }
        n_layers_ = n_layers;
        n_kv_heads_ = n_kv_heads;
        max_seq_len_ = max_seq_len;
        head_dim_ = head_dim;
        use_fp16_ = use_fp16;
        seq_len_ = 0;
        // 一层平面的元素数：n_kv_heads * max_seq_len * head_dim
        layer_stride_ = static_cast<size_t>(n_kv_heads) * max_seq_len * head_dim;
        // arena 总大小：2（K+V）* n_layers * layer_stride
        const size_t total = 2 * static_cast<size_t>(n_layers) * layer_stride_;
        if (use_fp16_) {
            data_f16_.assign(total, 0);           // fp16 arena 清零
        } else {
            data_.assign(total, 0.0f);            // fp32 arena 清零
        }
    }

    void KvCache::reset() {
        seq_len_ = 0;
    }

    // =========================================================================
    // k / v: 读取（仅 fp32 模式）。fp16 模式请用 k_f16()/v_f16()。
    // =========================================================================
    float *KvCache::k(int layer) {
        if (use_fp16_) {
            std::fprintf(stderr, "tinyqwen: KvCache::k() called in fp16 mode; use k_f16()\n");
            std::abort();
        }
        return data_.data() + static_cast<size_t>(layer) * layer_stride_;
    }
    float *KvCache::v(int layer) {
        if (use_fp16_) {
            std::fprintf(stderr, "tinyqwen: KvCache::v() called in fp16 mode; use v_f16()\n");
            std::abort();
        }
        return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }
    const float *KvCache::k(int layer) const {
        if (use_fp16_) {
            std::fprintf(stderr, "tinyqwen: KvCache::k() called in fp16 mode; use k_f16()\n");
            std::abort();
        }
        return data_.data() + static_cast<size_t>(layer) * layer_stride_;
    }
    const float *KvCache::v(int layer) const {
        if (use_fp16_) {
            std::fprintf(stderr, "tinyqwen: KvCache::v() called in fp16 mode; use v_f16()\n");
            std::abort();
        }
        return data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }

    // =========================================================================
    // k_f16 / v_f16: 读取（仅 fp16 模式），返回 fp16 指针。
    // 供 fp16-KV 融合 attention 直接读取（寄存器内转 fp32）。
    // =========================================================================
    uint16_t *KvCache::k_f16(int layer) {
        return data_f16_.data() + static_cast<size_t>(layer) * layer_stride_;
    }
    uint16_t *KvCache::v_f16(int layer) {
        return data_f16_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }
    const uint16_t *KvCache::k_f16(int layer) const {
        return data_f16_.data() + static_cast<size_t>(layer) * layer_stride_;
    }
    const uint16_t *KvCache::v_f16(int layer) const {
        return data_f16_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
    }

    // =========================================================================
    // write_token: 写入位置 pos 的 K/V（全部 kv 头），按存储精度转换。
    // =========================================================================
    // k_all / v_all: [n_kv_heads * head_dim] 连续（所有头），fp32。
    void KvCache::write_token(int layer, int pos, const float *k_all, const float *v_all) {
        const size_t head_plane = static_cast<size_t>(max_seq_len_) * head_dim_;
        const size_t pos_off = static_cast<size_t>(pos) * head_dim_;
        if (!use_fp16_) {
            // fp32: 逐头 memcpy
            float *k_layer = data_.data() + static_cast<size_t>(layer) * layer_stride_;
            float *v_layer =
                data_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
            for (int h = 0; h < n_kv_heads_; ++h) {
                std::memcpy(k_layer + h * head_plane + pos_off, k_all + h * head_dim_,
                            head_dim_ * sizeof(float));
                std::memcpy(v_layer + h * head_plane + pos_off, v_all + h * head_dim_,
                            head_dim_ * sizeof(float));
            }
        } else {
            // fp16: 逐头 fp32 -> fp16 转换后存储
            uint16_t *k_layer = data_f16_.data() + static_cast<size_t>(layer) * layer_stride_;
            uint16_t *v_layer =
                data_f16_.data() + (static_cast<size_t>(n_layers_) + layer) * layer_stride_;
            for (int h = 0; h < n_kv_heads_; ++h) {
                const float *kh = k_all + h * head_dim_;
                uint16_t *kdst = k_layer + h * head_plane + pos_off;
                for (int d = 0; d < head_dim_; ++d) kdst[d] = float_to_half(kh[d]);
                const float *vh = v_all + h * head_dim_;
                uint16_t *vdst = v_layer + h * head_plane + pos_off;
                for (int d = 0; d < head_dim_; ++d) vdst[d] = float_to_half(vh[d]);
            }
        }
    }

    void KvCache::advance(int n) {
        if (seq_len_ + n > max_seq_len_) {
            std::fprintf(stderr, "tinyqwen: KV cache overflow: seq_len %d + %d > max %d\n",
                         seq_len_, n, max_seq_len_);
            std::abort();
        }
        seq_len_ += n;
    }
} // namespace tinyqwen
