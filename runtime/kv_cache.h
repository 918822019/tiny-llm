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
//   - 全精度 fp32 存储（作为参考实现）
//   - 内存一次性分配，避免反复 new/delete 的碎片和延迟
// ============================================================================

#include <cstddef>
#include <vector>

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // KvCache: batch=1、decode 式追加的 K/V 缓存
    //
    // 内存布局（一整块 arena，先 K 后 V）:
    //   整体: [2 * n_layers * layer_stride] 个 float
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
    //
    // 版本说明:
    //   v1 实现已完成的功能: 追加写入、重置清空、容量检查
    //   v1 未实现的功能: speculative rollback（推测性回滚）；
    //     将来的扩展点: 加一个 truncate_to(new_len) 方法，当前布局不需要改
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
        // ---------------------------------------------------------------------
        KvCache(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

        // ---------------------------------------------------------------------
        // init: 一次性分配全部内存
        //
        // 参数:
        //   同构造函数参数
        //
        // 说明:
        //   采用 arena 分配策略: 一整块 std::vector<float> 容纳所有层、所有头
        //   的 K 和 V。这样做的好处是:
        //     - 仅一次分配，减少 malloc 调用
        //     - 内存连续，cache 友好
        //     - 释放时也是一次，不会产生碎片
        //   详见 .cpp 中的 arena 实现说明。
        // ---------------------------------------------------------------------
        void init(int n_layers, int n_kv_heads, int max_seq_len, int head_dim);

        // ---------------------------------------------------------------------
        // reset: 逻辑上清空缓存
        //
        // 说明:
        //   seq_len 归零，表示缓存中无有效数据。
        //   但保留已分配的内存，下次使用不需要重新分配。
        //   用于开始新的一段对话时重置状态。
        // ---------------------------------------------------------------------
        void reset();

        // ---- 属性访问器 ----

        // 当前已缓存了多少个 token 的 K/V
        int seq_len() const { return seq_len_; }

        // KV 缓存的最大容量（序列长度上限）
        int max_seq_len() const { return max_seq_len_; }

        // full attention 层的数量
        int n_layers() const { return n_layers_; }

        // 每层 key/value 注意力头的数量
        int n_kv_heads() const { return n_kv_heads_; }

        // 每个注意力头内部的维度大小
        int head_dim() const { return head_dim_; }

        // ---- 数据访问 ----

        // ---------------------------------------------------------------------
        // k: 获取第 layer 层的 K 缓存指针（可写版本）
        //
        // 参数:
        //   layer: 层索引（0-based）
        //
        // 返回值:
        //   指向该层 K 缓存的 float* 指针，布局为 [n_kv_heads][max_seq_len][head_dim]
        //
        // 说明:
        //   调用方拿到指针后，按上面的偏移公式计算具体位置的地址。
        // ---------------------------------------------------------------------
        float *k(int layer);

        // ---------------------------------------------------------------------
        // v: 获取第 layer 层的 V 缓存指针（可写版本）
        //
        // 参数:
        //   layer: 层索引（0-based）
        //
        // 返回值:
        //   指向该层 V 缓存的 float* 指针，布局为 [n_kv_heads][max_seq_len][head_dim]
        // ---------------------------------------------------------------------
        float *v(int layer);

        // ---------------------------------------------------------------------
        // k / v: 只读版本，用于 const 上下文
        // ---------------------------------------------------------------------
        const float *k(int layer) const;
        const float *v(int layer) const;

        // ---------------------------------------------------------------------
        // advance: 将 seq_len 向前推进 n 个位置
        //
        // 参数:
        //   n: 推进的 token 数（通常为 1，prefill 时可能 > 1）
        //
        // 说明:
        //   seq_len += n，表示刚才写入了 n 个新 token 的 K/V。
        //   如果 seq_len 超过 max_seq_len，直接 abort（fail fast，
        //   见 primer 第 9 节），表示这个序列太长了，当前 KV 缓存装不下。
        // ---------------------------------------------------------------------
        void advance(int n);

        // ---------------------------------------------------------------------
        // memory_bytes: 缓存占用的总字节数
        //
        // 返回值:
        //   整个 KV 缓存占用的内存字节数 = data_.size() * sizeof(float)
        //
        // 说明:
        //   用于打印内存占用信息，帮助用户了解资源消耗。
        // ---------------------------------------------------------------------
        size_t memory_bytes() const { return data_.size() * sizeof(float); }

    private:
        // 下面这些参数在 init() 时确定，之后不再改变
        int n_layers_ = 0;       // full attention 层数
        int n_kv_heads_ = 0;     // 每层 KV 头数
        int max_seq_len_ = 0;    // 最大序列长度
        int head_dim_ = 0;       // 每个头的维度
        int seq_len_ = 0;        // 当前已缓存的有效 token 数

        // 一层 K 或 V 占多少个 float 元素
        // = n_kv_heads_ * max_seq_len_ * head_dim_
        size_t layer_stride_ = 0;

        // 一整块 arena 内存
        // 总大小 = 2 * n_layers_ * layer_stride_ 个 float
        // 前半是 K（n_layers_ 层），后半是 V（n_layers_ 层）
        std::vector<float> data_;
    };
} // namespace tinyqwen