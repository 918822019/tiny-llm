#pragma once

// ============================================================================
// 文件: model_loader.h
// 作用: ModelFile 定义 —— 把 .tqwen 文件读进内存并校验，提供按名字取 tensor 的能力
//
// 核心职责:
//   1. 读取并校验 .tqwen 二进制权重文件的完整性
//   2. 解析文件头（TinyHeader）和 tensor 表（TensorEntry），构建 ModelConfig
//   3. 按名字提供 TensorView（不拥有数据的视图），供 QwenModel 绑定权重
//
// 二进制格式规范:
//   .tqwen 是 tiny-llm 自定义的扁平二进制格式，文件布局:
//     [ TinyHeader (192B) ][ TensorEntry[] (每项120B) ][ 64B对齐填充 ][ 权重数据区 ]
//   每条 tensor 数据起点都 64B 对齐，所有整数用小端字节序。
//
// 内存所有权:
//   ModelFile 把整个文件读进内部的 data_ buffer；所有 TensorView 的 data 指针
//   都指向 data_ 内部。因此 ModelFile 必须比所有使用这些指针的地方活得更久
//   （见 primer 第 3 节"视图"）。因为持有这块内存，所以禁止拷贝。
//
// 前置阅读: docs/infra_primer.md 第 5、9 节（二进制格式 / fail fast）
// ============================================================================

#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "tiny_format.h"

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // ModelConfig: 从文件头解析出来的模型配置
    //
    // 这个结构体描述"这是一个什么形状的模型"——它有多少层、隐藏层多宽、
    // 注意力头数量和维度、词表大小等。所有字段在 ModelFile::load() 时
    // 从 TinyHeader 中解析填充，之后不再改变。
    //
    // 支持两种架构:
    //   - Qwen2.x: 所有层同构（full attention + SwiGLU）
    //   - Qwen3.5: 混合架构（Gated DeltaNet + full attention 按间隔交替）
    // -------------------------------------------------------------------------
    struct ModelConfig {
        uint32_t n_layers = 0;          // transformer 层数
        uint32_t hidden_size = 0;       // 隐藏层宽度（每个 token 的向量维度）
        uint32_t intermediate_size = 0; // FFN 中间层宽度（通常是 hidden_size 的倍数）
        uint32_t n_heads = 0;           // query 注意力头数
        uint32_t n_kv_heads = 0;        // key/value 头数（GQA，通常 ≤ n_heads）
        uint32_t head_dim = 0;          // 每个注意力头的维度
        uint32_t vocab_size = 0;        // 词表大小
        uint32_t max_seq_len = 0;       // 训练时支持的最大序列长度
        float rms_norm_eps = 0.0f;      // RMSNorm 中防止除零的小常数
        float rope_theta = 0.0f;        // RoPE 位置编码的底数 theta
        bool tied_embeddings = false;   // 是否共享词嵌入和输出投影的权重

        // --- v2 扩展字段（v1 文件加载后保持默认值，行为与旧版完全一致）---
        ModelType model_type = ModelType::kQwen2; // 模型架构族

        // Qwen3.5 线性注意力（Gated DeltaNet）层的形状参数
        uint32_t linear_num_qk_heads = 0;    // GDN 线性注意力的 Q/K 头数
        uint32_t linear_num_v_heads = 0;     // GDN 线性注意力的 V 头数
        uint32_t linear_qk_head_dim = 0;     // GDN 每个 Q/K 头的维度
        uint32_t linear_v_head_dim = 0;      // GDN 每个 V 头的维度
        uint32_t linear_conv_kernel_dim = 0; // GDN 内 causal conv1d 的 kernel 大小

        // full attention 层的出现间隔: 当 (layer_idx + 1) % full_attention_interval == 0
        // 时该层为 full attention，否则为 linear attention（GDN）。
        // 0 表示全部是 full attention（v1 行为）。
        uint32_t full_attention_interval = 0;

        // RoPE 只旋转 head_dim 的这一比例（例如 Qwen3.5 为 0.25）
        float partial_rotary_factor = 1.0f;

        // 结束符 token id（Qwen3.5 = 248044），0 表示未指定，用 CLI 默认
        uint32_t eos_token_id = 0;

        // --- 量化参数（仅 dtype == kI4 时有意义）---
        uint32_t quant_group_size = 0; // 0 = 未量化；128 = INT4 典型 group size

        // --- v3 / MoE 扩展字段（仅 ModelType == kQwen35MoE 时有意义）---
        uint32_t n_routed_experts = 0;             // 路由专家数（如 256；fake 用小值）
        uint32_t num_experts_per_tok = 0;            // 每个 token 激活的路由专家数 k
        uint32_t moe_intermediate_size = 0;          // 每个路由专家 FFN 的中间层宽度
        uint32_t shared_expert_intermediate_size = 0;// 共享专家 FFN 的中间层宽度
        uint32_t n_shared_experts = 0;               // 共享专家数（通常 1）
        uint32_t moe_topk_norm = 1;                  // 1 = top-k 权重 softmax 归一
        uint32_t gptq_group_size = 0;                // GPTQ 每组元素数（专家为 kGPTQ4 时 >0）

        // 是否为 MoE 架构（FFN 换成 MoE，attention 部分与 kQwen35 完全一致）
        bool is_moe() const {
            return model_type == ModelType::kQwen35MoE ||
                   model_type == ModelType::kQwen3MoE;
        }

        // MoE 是否有共享专家（Qwen3-MoE 没有，Qwen3.5-MoE 有）
        bool has_shared_expert() const { return n_shared_experts > 0; }

        // attention 是否走 Qwen3.5 风格（q_proj 携带输出门故形状为 [2*q_dim, hidden]、
        // partial RoPE、GDN 混合）。权重绑定与前向计算**必须用同一个判据**，否则
        // 绑定的形状与计算不符。历史上 create() 用 (kQwen35 || is_moe()) 而
        // forward 用 (kQwen35 || kQwen35MoE)，对 kQwen3MoE 取值不同——它虽是 MoE
        // 但 attention 是普通 Qwen3 dense 风格（q_proj [q_dim, hidden] + 可选
        // QK-norm），必须走非 qwen35 分支，故这里不含 kQwen3MoE。
        bool uses_qwen35_attention() const {
            return model_type == ModelType::kQwen35 ||
                   model_type == ModelType::kQwen35MoE;
        }

        // ---------------------------------------------------------------------
        // is_linear_layer: 判断 layer_idx 是否为 linear attention（GDN）层
        //
        // 参数:
        //   layer_idx: 层索引（0-based）
        //
        // 返回值:
        //   当 full_attention_interval <= 1（v1 行为）或 model_type == kQwen2 时
        //   恒为 false；否则按 full_attention_interval 判断。
        // ---------------------------------------------------------------------
        bool is_linear_layer(uint32_t layer_idx) const {
            // interval <= 1 意味着所有层都是 full attention，没有线性层
            if (full_attention_interval <= 1) return false;
            // 线性层: 满足 (layer_idx+1) 不能被 interval 整除
            return (layer_idx + 1) % full_attention_interval != 0;
        }

        // ---------------------------------------------------------------------
        // n_full_layers: 计算 full attention 层的数量
        //
        // 返回值:
        //   interval <= 1 时所有层都是 full attention，返回 n_layers；
        //   否则返回 n_layers / full_attention_interval（整数除法取整）。
        // ---------------------------------------------------------------------
        int n_full_layers() const {
            if (full_attention_interval <= 1) return static_cast<int>(n_layers);
            return static_cast<int>(n_layers / full_attention_interval);
        }

        // ---------------------------------------------------------------------
        // full_layer_cache_index: 将全局层号映射到 KV cache 中的紧凑下标
        //
        // 参数:
        //   layer_idx: 全局层索引（0-based）
        //
        // 返回值:
        //   该层在 KV cache 数组中的位置（0-based）
        //
        // 说明:
        //   只有 full attention 层才分配 KV cache，所以需要把全局层号
        //   压缩映射到 KV cache 的紧凑下标。例如 interval=4 时:
        //     layer 3、7、11... 映射到 cache index 0、1、2...
        // ---------------------------------------------------------------------
        int full_layer_cache_index(uint32_t layer_idx) const {
            // interval <= 1 = 所有层都是 full attention（如 kQwen3MoE），此时全局层号
            // 就是紧凑下标。缺这个守卫会除零——is_linear_layer / n_full_layers 都有
            // <= 1 守卫，本函数原先没有（同结构体三个函数守卫不一致）。
            if (full_attention_interval <= 1) return static_cast<int>(layer_idx);
            return static_cast<int>((layer_idx + 1) / full_attention_interval - 1);
        }

        // ---------------------------------------------------------------------
        // linear_layer_cache_index: 将全局层号映射到 GDN 状态中的紧凑下标
        //
        // 参数:
        //   layer_idx: 全局层索引（0-based）
        //
        // 返回值:
        //   该层在 GDN 状态数组中的位置（0-based）
        //
        // 说明:
        //   线性层在 GDN 状态数组里是紧凑排列的，需要跳过前面的 full attention 层。
        //   公式: 全局层号 - 前面的 full attention 层数
        // ---------------------------------------------------------------------
        int linear_layer_cache_index(uint32_t layer_idx) const {
            return static_cast<int>(layer_idx - (layer_idx + 1) / full_attention_interval);
        }
    };

    // -------------------------------------------------------------------------
    // ModelFile: 加载并校验一个 .tqwen 文件
    //
    // 内存所有权:
    //   本对象把整个文件读进自己内部的 data_ buffer；所有 TensorView 的 data
    //   指针都指向 data_ 内部。因此 ModelFile 必须比所有使用这些指针的地方
    //   活得更久。因为持有这块内存，所以禁止拷贝（避免误拷贝整块权重内存）。
    //
    // 加载流程:
    //   1. 读取整个文件到 data_ buffer
    //   2. 校验魔数 "TINYQWEN" 和版本号
    //   3. 解析 TinyHeader，填充 ModelConfig
    //   4. 解析 tensor 表，为每个 tensor 创建 TensorView
    //   5. 校验每个 tensor 的偏移和大小在文件范围内
    // -------------------------------------------------------------------------
    class ModelFile {
    public:
        ModelFile() = default;

        // 禁止拷贝（避免误拷贝整块权重内存，可能导致 OOM 或 double-free）
        ModelFile(const ModelFile &) = delete;
        ModelFile &operator=(const ModelFile &) = delete;

        // ---------------------------------------------------------------------
        // load: 读取并校验整个 .tqwen 文件
        //
        // 参数:
        //   path: .tqwen 文件的路径
        //   err:  输出参数，任何一步失败都返回 false 并把原因写进 *err
        //
        // 返回值:
        //   成功返回 true，失败返回 false
        //
        // 校验项目:
        //   - 文件是否存在、是否能打开
        //   - 魔数是否为 "TINYQWEN"
        //   - 版本号是否在 [kFormatVersionMin, kFormatVersion] 范围内
        //   - tensor 表和数据区偏移是否在文件范围内
        //   - 每个 tensor 的 offset + nbytes 是否在文件范围内
        //   - tensor 的 shape 和 dtype 是否合法
        //
        // offload_experts:
        //   false（默认）= 整文件读进 data_，行为与历史版本逐字节一致。
        //   true  = 稀疏加载：只把 resident tensor 紧凑打包进 data_，名字含
        //           ".mlp.experts." 的路由专家 tensor 一个字节都不读，只记
        //           file_offset/nbytes（其 TensorView.data 为 nullptr），
        //           由 ExpertStore 按需 pread。这是 MoE SSD 卸载真正省内存的
        //           前提——不开这个开关，专家权重照样全量驻留 RAM。
        // ---------------------------------------------------------------------
        bool load(const std::string &path, std::string *err, bool offload_experts = false);

        // ---------------------------------------------------------------------
        // estimate_resident_bytes: 预检——算出 load() 之后会占多少常驻内存
        //
        // 为什么需要它：load() 会把权重真正读进 RAM。内存预算校验若放在 load()
        // 之后，等校验失败时文件已经进内存了——先犯罪再审查，且换页已经发生。
        // 这个函数只读 header（192 字节）+ tensor 表，**一个数据区字节都不读**，
        // 因此能在 load() 之前就把内存需求算准，让调用方 fail-fast。
        //
        // 参数:
        //   path:            .tqwen 文件路径
        //   offload_experts: 与 load() 的同名参数语义一致
        //                    false = 整文件进 RAM，返回值 == 文件大小
        //                    true  = 稀疏加载，返回值 == 紧凑打包后的 data_ 大小
        //   out_bytes:       输出参数，成功时写入字节数
        //   err:             输出参数，失败时写入原因
        //
        // 返回值: 成功 true，失败 false
        //
        // 口径保证：返回值与随后真正调用 load(path, err, offload_experts) 得到的
        // resident_bytes() **必须一致**。两处共用 is_offloadable() 与 align_up()，
        // 判据是名字含 ".mlp.experts."（外加 tied 模型不可卸载 embed 的守卫）。
        // ---------------------------------------------------------------------
        static bool estimate_resident_bytes(const std::string &path, bool offload_experts,
                                            uint64_t *out_bytes, std::string *err);

        // ---- 状态查询 ----

        // 是否已成功加载文件
        bool loaded() const { return !data_.empty(); }

        // 文件数据在内存中的起始地址（用于计算 tensor 的绝对地址）
        // 稀疏加载下这是紧凑缓冲的起点，不等于文件起点
        const uint8_t *base() const { return data_.data(); }

        // ---- 内存归因（稀疏加载才区分 resident / offloaded）----

        // 磁盘上 .tqwen 的真实总字节数
        size_t file_bytes() const { return file_bytes_; }

        // 实际驻留 RAM 的字节数（data_ 的大小）
        size_t resident_bytes() const { return data_.size(); }

        // 留盘未读的专家权重字节数（稀疏加载下 > 0；全量加载恒为 0）
        size_t offloaded_bytes() const { return offloaded_bytes_; }

        // 被卸载的 tensor 个数
        size_t offloaded_count() const { return offloaded_count_; }

        // 获取文件头（包含魔数、版本、配置等）
        const TinyHeader &header() const { return header_; }

        // 获取模型配置（从文件头解析出来的结构体）
        const ModelConfig &config() const { return config_; }

        // ---- 张量访问 ----

        // ---------------------------------------------------------------------
        // get: 按名字获取 tensor 的只读视图
        //
        // 参数:
        //   name: tensor 名字，如 "model.layers.0.mlp.up_proj.weight"
        //
        // 返回值:
        //   指向 TensorView 的指针；如果名字不存在返回 nullptr
        //
        // 说明:
        //   返回的 TensorView 的 data 指针指向 data_ 内部，不拷贝数据。
        // ---------------------------------------------------------------------
        const TensorView *get(const std::string &name) const;

        // 文件中 tensor 的总数
        size_t tensor_count() const { return order_.size(); }

        // 所有 tensor 名字的列表（按文件内出现顺序排列）
        const std::vector<std::string> &tensor_names() const { return order_; }

        // ---------------------------------------------------------------------
        // print_summary: 打印模型摘要信息
        //
        // 输出内容:
        //   - 模型配置（层数、隐藏层维度、头数等）
        //   - 每个 tensor 的名字、形状、dtype 和偏移
        //
        // 说明:
        //   调试用，帮助快速了解模型文件的内容和结构。
        // ---------------------------------------------------------------------
        void print_summary() const;

    private:
        // 整个文件的内容（我们拥有的唯一一份内存，所有 TensorView 的 data 都指向这里）
        // 稀疏加载时只含 header+表+resident tensor
        std::vector<uint8_t> data_;

        // 文件头（192 字节），包含魔数、版本、模型配置等
        TinyHeader header_{};

        // 从文件头解析出的模型配置结构体
        ModelConfig config_{};

        // 名字到 TensorView 的映射表，O(1) 平均查找时间
        std::unordered_map<std::string, TensorView> tensors_;

        // 按文件内顺序记录 tensor 名字，供 print_summary() 按序打印
        std::vector<std::string> order_;

        // 磁盘文件真实大小（稀疏加载下 data_.size() < file_bytes_）
        size_t file_bytes_ = 0;

        // 留盘未读的专家权重字节数与 tensor 个数
        size_t offloaded_bytes_ = 0;
        size_t offloaded_count_ = 0;
    };
} // namespace tinyqwen