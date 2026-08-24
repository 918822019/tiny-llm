#pragma once

// ============================================================================
// 文件: kv_cache.h
// 作用: KV cache 实现 —— decode 阶段的记忆存储
//
// 为什么需要 KV Cache?
//   attention 里每个新 token 都要看之前所有 token 的 K 和 V 向量。如果每生成
//   一个词都重算前面所有词的 K/V，计算量会平方级爆炸（O(n^2)）。所以把历史
//   token 算出的 K/V 向量缓存起来复用 —— 这就是 KV cache。
//
// 前置阅读: docs/infra_primer.md 第 3、6 节
//
// 设计约束:
//   - batch = 1（单条序列推理）
//   - decode 式追加写入（每次前进 1 个位置）
//   - 内存一次性分配，避免反复 new/delete 的碎片和延迟
//
// 存储精度（v2 新增）:
//   - fp32（默认，参考实现）：全精度存储，作为正确性基准。
//   - fp16（use_fp16=true，opt-in）：K/V 以 fp16 存储，内存与 attention
//     读取带宽减半（长上下文收益）。写入时 fp32→fp16，读取（attention）时
//     反量化回 fp32 到一个 workspace，attention 计算仍在 fp32 进行——
//     即"存 fp16、算 fp32"，质量损失可忽略，attention kernel 无需改动。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // KvCache: batch=1、decode 式追加的 K/V 缓存
    //
    // 内存布局（一整块 arena，先 K 后 V）:
    //   整体: [2 * n_layers * layer_stride] 个元素（元素类型随精度而变）
    //   K 区: [n_layers][n_kv_heads][max_seq_len][head_dim]
    //   V 区: [n_layers][n_kv_heads][max_seq_len][head_dim]
    //
    // 读法:
    //   第 1 维是层号，第 2 维是 kv 头号，第 3 维是序列位置，第 4 维是头内维度
    //
    // 访问公式:
    //   访问某层 l、某头 h、某位置 t 的起点偏移（元素个数）:
    //     K: l * layer_stride_ + h * (max_seq_len * head_dim) + t * head_dim
    //     V: n_layers * layer_stride_ + l * layer_stride_ + ...（同上）
    // -------------------------------------------------------------------------
    class KvCache {
    public:
        // 默认构造: 不分配任何内存，需后续调用 init() 初始化
        KvCache() = default;

        // ---------------------------------------------------------------------
        // 带参数构造: 直接构造并分配内存
        //
        // 参数:
        //   n_layers:    transformer 层数（仅 full attention 层）
        //   n_kv_heads:  每层 key/value 的注意力头数（GQA）
        //   max_seq_len: 最大序列长度（KV 容量上限）
        //   head_dim:    每个注意力头的维度
        //   use_fp16:    是否用 fp16 存储（默认 false = fp32 参考）
        // ---------------------------------------------------------------------
        KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim,
                bool use_fp16 = false);

        // ---------------------------------------------------------------------
        // init: 一次性分配全部内存
        //
        // 说明:
        //   采用 arena 分配策略: 一整块连续内存容纳所有层、所有头的 K 和 V。
        //   use_fp16=true 时分配 uint16_t（fp16）arena + 一个 fp32 反量化
        //   workspace（读时把 fp16 反量化回 fp32 供 attention 用）。
        // ---------------------------------------------------------------------
        void init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim,
                  bool use_fp16 = false);

        // ---------------------------------------------------------------------
        // reset: 逻辑上清空缓存（seq_len 归零，保留内存）
        // ---------------------------------------------------------------------
        void reset();

        // ---- 属性访问器 ----
        int seq_len() const { return seq_len_; }
        int max_seq_len() const { return max_seq_len_; }
        int n_layers() const { return n_layers_; }
        int n_kv_heads() const { return n_kv_heads_; }
        int head_dim() const { return head_dim_; }
        bool use_fp16() const { return use_fp16_; }

        // ---- 数据访问（读）----
        // ---------------------------------------------------------------------
        // k / v: 获取第 layer 层的 K/V 缓存（仅 fp32 模式；供 fp32 attention 读）
        //
        // 返回值: 指向该层 K/V 的 float* 指针，布局 [n_kv_heads][max_seq_len][head_dim]
        //
        // fp16 模式下不应调用本方法（存储是 fp16），请用 k_f16()/v_f16()。
        // ---------------------------------------------------------------------
        float *k(int layer);
        float *v(int layer);
        const float *k(int layer) const;
        const float *v(int layer) const;

        // ---------------------------------------------------------------------
        // k_f16 / v_f16: fp16 模式下获取第 layer 层的 K/V 缓存（fp16 指针）
        //
        // 返回值: 指向该层 K/V 的 uint16_t*（fp16）指针，
        //   布局 [n_kv_heads][max_seq_len][head_dim]。
        // 供 fp16-KV 融合 attention 直接读取（寄存器内转 fp32），
        // 避免独立的反量化遍历。仅 fp16 模式有效。
        // ---------------------------------------------------------------------
        uint16_t *k_f16(int layer);
        uint16_t *v_f16(int layer);
        const uint16_t *k_f16(int layer) const;
        const uint16_t *v_f16(int layer) const;

        // ---- 数据访问（写）----
        // ---------------------------------------------------------------------
        // write_token: 写入某个位置 pos 的 K/V（全部 kv 头）
        //
        // 参数:
        //   layer:  层索引
        //   pos:    序列位置（通常 = 当前 seq_len_）
        //   k_all:  该 token 的 K，[n_kv_heads * head_dim] 连续（所有头），fp32
        //   v_all:  该 token 的 V，[n_kv_heads * head_dim] 连续（所有头），fp32
        //
        // 说明: 统一写入入口，内部按存储精度转换（fp32 直接 memcpy；fp16 先
        //   fp32→fp16 再存）。调用方不再直接 memcpy 到 k()/v() 指针。
        // ---------------------------------------------------------------------
        void write_token(int layer, int pos, const float *k_all, const float *v_all);

        // ---------------------------------------------------------------------
        // advance: 将 seq_len 向前推进 n 个位置（超限 abort）
        // ---------------------------------------------------------------------
        void advance(int n);

        // ---------------------------------------------------------------------
        // memory_bytes: 缓存占用的总字节数（不含反量化 workspace）
        // ---------------------------------------------------------------------
        size_t memory_bytes() const {
            return use_fp16_ ? data_f16_.size() * sizeof(uint16_t)
                             : data_.size() * sizeof(float);
        }

    private:
        int n_layers_ = 0;
        int n_kv_heads_ = 0;
        int max_seq_len_ = 0;
        int head_dim_ = 0;
        int seq_len_ = 0;
        bool use_fp16_ = false;     // 存储精度开关

        // 一层 K 或 V 占多少个元素 = n_kv_heads_ * max_seq_len_ * head_dim_
        size_t layer_stride_ = 0;

        // fp32 arena（use_fp16_=false 时用）
        std::vector<float> data_;
        // fp16 arena（use_fp16_=true 时用）。attention 直接读 fp16、
        // 寄存器内转 fp32（融合反量化），不做独立反量化遍历。
        std::vector<uint16_t> data_f16_;
    };
} // namespace tinyqwen
