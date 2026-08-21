// ============================================================================
// qwen_model.cpp — Qwen 模型核心实现（模型创建和权重绑定）
// ============================================================================
// 本文件实现 QwenModel 类，这是整个 tiny-llm 推理引擎的核心。包含：
//   1. QwenModel::create() 工厂函数：从 .tqwen 模型文件加载权重，校验
//      每个 tensor 的 shape/dtype，分配 KV cache 和 workspace 缓冲区
//   2. 权重绑定辅助函数：require_view() 和 bind_f32_vector()，确保每个
//      权重 tensor 都通过严格校验后才被使用
//   3. 矩阵乘法封装函数：mv() / mv_pair() / mv_qkv() / mm()，通过 backend
//      接口调用具体实现
//
// 一个 token 的前向大致是：
//   查词嵌入 -> 逐层 [attention + FFN] -> 最后 norm -> 投影到词表得 logits -> argmax
//
// 刻意不做任何图抽象：forward 就是一个可读的函数，op 顺序与文档一一对应，
// 跟 PyTorch 出现数值偏差时肉眼可查。
//
// workspace buffer（create() 时一次分配，每个 token 不再分配）：
//   hidden_  残差流：贯穿所有层的主数据，残差不断往上加
//   normed_  每次 RMSNorm 的输出，喂给接下来的投影
//   q_/k_/v_ attention 的 query/key/value 投影结果
//   attn_    attention 输出（各 head 拼接）
//   o_       o_proj 输出
//   gate_/up_/ffn_  SwiGLU FFN 的中间量
//   logits_  lm_head 输出，对词表每个词一个分数
// ============================================================================

#include "qwen_model.h"

#include <algorithm>   // std::min, std::partial_sort
#include <cmath>       // std::sqrt, std::exp, std::log1p
#include <cstdio>      // 标准输入输出
#include <cstring>     // 内存操作
#include <numeric>     // std::iota

#include "backend_cpu.h"  // create_cpu_backend
#include "dispatch.h"     // matvec_f32 通用入口（分发到 _ref / 将来的优化版）
#include "gdn_ops.h"      // Qwen3.5 GDN 算子（l2norm / conv1d / delta rule / ...）
#include "ref_ops.h"

namespace tinyqwen {
    namespace {
        // =====================================================================
        // quant_type_of() — Dtype -> QuantType 的安全映射
        // =====================================================================
        // 文件格式 Dtype 与后端 QuantType 数值不一致（Dtype 预留 kI8=2、kI4=3；
        // QuantType 为 kF32=0/kF16=1/kI4=2），直接 static_cast 会把 i4 权重错误
        // 路由到 f32 matvec，把打包字节当 float 读导致越界崩溃，必须显式映射。
        QuantType quant_type_of(Dtype d) {
            switch (d) {
                case Dtype::kF16: return QuantType::kF16;
                case Dtype::kI4:  return QuantType::kI4;
                default:          return QuantType::kF32;
            }
        }

        // =====================================================================
        // top_k_logits() — 取分数最高的 k 个 token
        // =====================================================================
        // 参数：
        //   logits — logits 分数数组 [vocab]
        //   vocab  — 词表大小
        //   k      — 要取的最大值数量
        //   out    — 输出结果（indices 和 values 从大到小排列）
        // 说明：实现使用 partial_sort，只把前 k 个排好序，复杂度 O(vocab * k)。
        //       v1 够用（每 token 调一次，不在 kernel 热点路径上）。
        //       结果第一个就是 argmax。
        void top_k_logits(const float *logits, int vocab, int k, TopKResult *out) {
            k = std::min(k, vocab); // k 不能超过词表大小
            // 创建索引数组 [0, 1, 2, ..., vocab-1]
            std::vector<int> idx(vocab);
            std::iota(idx.begin(), idx.end(), 0);
            // 部分排序：只把前 k 个最大值的索引排到前面
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [logits](int a, int b) { return logits[a] > logits[b]; });
            // 输出前 k 个
            out->indices.resize(k);
            out->values.resize(k);
            for (int i = 0; i < k; ++i) {
                out->indices[i] = idx[i];
                out->values[i] = logits[idx[i]];
            }
        }

        // =====================================================================
        // shape_str() — 将张量形状转成可读字符串
        // =====================================================================
        std::string shape_str(const std::vector<uint64_t> &s) {
            std::string r = "[";
            for (size_t i = 0; i < s.size(); ++i) {
                if (i) r += ", ";
                r += std::to_string(s[i]);
            }
            return r + "]";
        }
    } // namespace

    // =========================================================================
    // QwenModel::require_view() — 按 HF 名字取 tensor 并校验 shape/dtype
    // =========================================================================
    // 参数：
    //   file  — 模型文件
    //   name  — tensor 名称（如 "model.layers.0.self_attn.q_proj.weight"）
    //   shape — 期望的形状向量
    //   err   — 输出参数，出错时写入错误信息
    // 返回值：合法时返回 TensorView 指针，否则返回 nullptr
    // 说明：shape 不对（比如导出了非 Qwen 的 checkpoint）在这里就变成明确报错，
    //       而不是后面悄悄算错（fail fast，见 primer 第 9 节）。
    //       dtype 必须等于文件级 dtype（v1 单一 dtype，loader 已保证；这里双保险）。
    const TensorView *QwenModel::require_view(const ModelFile &file, const std::string &name,
                                              const std::vector<uint64_t> &shape,
                                              std::string *err) {
        // 查找 tensor
        const TensorView *t = file.get(name);
        if (!t) {
            if (err) *err = "missing tensor: " + name;
            return nullptr;
        }
        // dtype 校验：I4 文件允许混合 dtype（大矩阵 kI4、小向量 kF32）
        if (dtype_ == Dtype::kI4) {
            if (t->dtype != Dtype::kI4 && t->dtype != Dtype::kF32) {
                if (err)
                    *err = "tensor " + name + " dtype mismatch: i4 file allows i4/f32, got " +
                           dtype_name(t->dtype);
                return nullptr;
            }
        } else if (t->dtype != dtype_) {
            // f32/f16 文件要求全文件单一 dtype
            if (err)
                *err = "tensor " + name + " dtype mismatch: got " + dtype_name(t->dtype) +
                       " expected " + dtype_name(dtype_);
            return nullptr;
        }
        // ndim 校验：维度数必须匹配
        if (static_cast<size_t>(t->ndim) != shape.size()) {
            if (err)
                *err = "tensor " + name + " ndim mismatch: got " + std::to_string(t->ndim) +
                       " expected " + std::to_string(shape.size());
            return nullptr;
        }
        // shape 逐维校验：每一维的大小必须匹配
        for (size_t i = 0; i < shape.size(); ++i) {
            if (t->shape[i] != shape[i]) {
                if (err) {
                    const std::vector<uint64_t> got_shape(t->shape, t->shape + t->ndim);
                    *err = "tensor " + name + " shape mismatch: got " + shape_str(got_shape) +
                           " expected " + shape_str(shape);
                }
                return nullptr;
            }
        }
        return t;
    }

    // =========================================================================
    // QwenModel::bind_f32_vector() — 将小向量绑定为 float32 指针
    // =========================================================================
    // 参数：
    //   t — TensorView 指针（已通过 require_view 校验）
    // 返回值：float32 数据指针
    // 说明：norm/bias 这类小向量恒按 fp32 使用：
    //       - f32 模型：零拷贝直指文件内存
    //       - f16 模型：在 create 期一次性转成 fp32 副本（总量 ~400KB 级，
    //         换算耗时可忽略），换来 rmsnorm/bias 的每 token 路径不需要感知 dtype
    //       - I4 模型：小向量存为 kF32，同样零拷贝
    const float *QwenModel::bind_f32_vector(const TensorView *t) {
        // f32 类型或 I4 文件中的小向量：零拷贝直指
        if (dtype_ == Dtype::kF32 || t->dtype == Dtype::kF32) return t->f32();
        // f16 模型：逐元素转换成 fp32 副本
        std::vector<float> buf(t->numel());
        const uint8_t *src = t->data;
        for (size_t i = 0; i < buf.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, src + i * 2, sizeof(h)); // 从 f16 字节流中取出一个 half
            buf[i] = half_to_float(h);               // 转为 fp32
        }
        owned_f32_.push_back(std::move(buf)); // 保存副本（所有权转移）
        return owned_f32_.back().data();      // 返回副本的指针
    }

    // =========================================================================
    // QwenModel::mv() — 矩阵-向量乘法（matvec）分派薄封装
    // =========================================================================
    // 参数：
    //   w       — 权重数据指针（void*，实际类型由 dtype_ 决定）
    //   x       — 输入向量 [in_dim]
    //   y       — 输出向量 [out_dim]
    //   out_dim — 输出维度（矩阵行数）
    //   in_dim  — 输入维度（矩阵列数）
    // 说明：通过 backend 接口调用具体实现，屏蔽 dtype 和实现细节。
    void QwenModel::mv(const void *w, const float *x, float *y, int out_dim, int in_dim) const {
        WeightTensor wt{w, quant_type_of(dtype_), out_dim, in_dim, group_size_};
        backend_->matvec(wt, x, y, out_dim, in_dim);
    }

    // =========================================================================
    // QwenModel::mv_pair() — 双输出矩阵-向量乘法（gate/up 融合）
    // =========================================================================
    // 参数：
    //   w1, w2  — 两个权重矩阵的数据指针
    //   x        — 输入向量 [in_dim]
    //   y1, y2   — 输出向量 [out_dim]
    //   out_dim, in_dim — 维度参数
    // 说明：同时计算 y1=W1*x 和 y2=W2*x，x 只加载一次。
    void QwenModel::mv_pair(const void *w1, const void *w2, const float *x, float *y1,
                            float *y2, int out_dim, int in_dim) const {
        WeightTensor wt1{w1, quant_type_of(dtype_), out_dim, in_dim, group_size_};
        WeightTensor wt2{w2, quant_type_of(dtype_), out_dim, in_dim, group_size_};
        backend_->matvec_pair(wt1, wt2, x, y1, y2, out_dim, in_dim);
    }

    // =========================================================================
    // QwenModel::mv_qkv() — 三输出矩阵-向量乘法（QKV 融合）
    // =========================================================================
    // 参数：
    //   wq, wk, wv — Q/K/V 三个权重矩阵的数据指针
    //   x           — 输入向量 [in_dim]
    //   yq, yk, yv  — Q/K/V 输出向量
    //   q_dim       — Q 输出维度
    //   kv_dim      — K/V 输出维度
    //   in_dim      — 输入维度
    // 说明：同时计算三个投影，x 只加载一次。
    void QwenModel::mv_qkv(const void *wq, const void *wk, const void *wv, const float *x,
                           float *yq, float *yk, float *yv, int q_dim, int kv_dim,
                           int in_dim) const {
        WeightTensor wqt{wq, quant_type_of(dtype_), q_dim, in_dim, group_size_};
        WeightTensor wkt{wk, quant_type_of(dtype_), kv_dim, in_dim, group_size_};
        WeightTensor wvt{wv, quant_type_of(dtype_), kv_dim, in_dim, group_size_};
        backend_->matvec_qkv(wqt, wkt, wvt, x, yq, yk, yv, q_dim, kv_dim, in_dim);
    }

    // =========================================================================
    // QwenModel::create() — 工厂函数：创建并初始化 QwenModel 实例
    // =========================================================================
    // 参数：
    //   file        — 已加载的模型文件
    //   max_seq_len — 最大序列长度（KV cache 容量）
    //   profiler    — 性能剖析器引用
    //   err         — 输出参数，出错时写入错误信息
    //   out         — 输出参数，成功时指向创建的模型实例
    //   backend     — 计算后端（可选，默认 CPU）
    // 返回值：成功返回 true，失败返回 false
    // 说明：这是模型的初始化函数，负责：
    //   1. 校验模型配置的合理性
    //   2. 绑定所有权重 tensor（按层遍历）
    //   3. 初始化 KV cache 和 GDN 状态
    //   4. 分配 workspace 缓冲区
    //   所有失败经 *err 报告；成功后 *out 持有一个可直接运行的模型。
    bool QwenModel::create(const ModelFile &file, int max_seq_len, Profiler &profiler,
                           std::string *err, std::unique_ptr<QwenModel> *out,
                           std::unique_ptr<IBackend> backend) {
        out->reset(); // 清空输出指针
        if (!file.loaded()) {
            if (err) *err = "model file not loaded";
            return false;
        }
        const ModelConfig &cfg = file.config();

        // max_seq_len <= 0 表示"用模型训练时的上限"
        if (max_seq_len <= 0) max_seq_len = static_cast<int>(cfg.max_seq_len);
        if (max_seq_len > static_cast<int>(cfg.max_seq_len)) {
            if (err)
                *err = "max_seq_len " + std::to_string(max_seq_len) +
                       " exceeds model max " + std::to_string(cfg.max_seq_len);
            return false;
        }

        // 构造模型实例
        std::unique_ptr<QwenModel> m(new QwenModel());
        m->cfg_ = cfg;
        m->dtype_ = static_cast<Dtype>(file.header().dtype); // 文件级 dtype
        m->group_size_ = static_cast<int>(cfg.quant_group_size); // INT4 量化组大小
        m->profiler_ = &profiler;
        m->max_seq_len_ = max_seq_len;
        // 创建后端：默认 CPU，未来可扩展 CUDA/Metal
        m->backend_ = backend ? std::move(backend) : create_cpu_backend();
        // Q 和 KV 的总维度
        m->q_dim_ = static_cast<int>(cfg.n_heads * cfg.head_dim);
        m->kv_dim_ = static_cast<int>(cfg.n_kv_heads * cfg.head_dim);

        // Qwen3.5 混合架构：预计算 GDN 和 partial RoPE 的派生维度
        const bool is_qwen35 = cfg.model_type == ModelType::kQwen35;
        if (is_qwen35) {
            // GDN linear attention 的维度
            m->gdn_qk_dim_ = static_cast<int>(cfg.linear_num_qk_heads * cfg.linear_qk_head_dim);
            m->gdn_value_dim_ =
                    static_cast<int>(cfg.linear_num_v_heads * cfg.linear_v_head_dim);
            // 卷积输入维度 = 2*qk_dim + v_dim（q、k 各一份，v 一份）
            m->gdn_conv_dim_ = 2 * m->gdn_qk_dim_ + m->gdn_value_dim_;
            // partial RoPE 实际旋转的维度
            m->rotary_dim_ =
                    static_cast<int>(std::lround(cfg.head_dim * cfg.partial_rotary_factor));
            // rotary_dim 必须是偶数（RoPE 的旋转对操作要求）
            if (m->rotary_dim_ % 2 != 0 || m->rotary_dim_ > static_cast<int>(cfg.head_dim)) {
                if (err)
                    *err = "partial_rotary_factor yields odd/oversized rotary_dim: " +
                           std::to_string(m->rotary_dim_);
                return false;
            }
        }

        const uint64_t hidden = cfg.hidden_size;      // 残差流维度
        const uint64_t inter = cfg.intermediate_size;  // FFN 中间维度
        const uint64_t vocab = cfg.vocab_size;         // 词表大小
        const uint64_t qd = static_cast<uint64_t>(m->q_dim_);
        const uint64_t kvd = static_cast<uint64_t>(m->kv_dim_);

        // 绑定辅助：大矩阵取裸字节指针（dtype 随模型），小向量恒绑 fp32
        const auto bind_mat = [&](const char *name, std::vector<uint64_t> shape,
                                  const void **out) -> bool {
            const TensorView *t = m->require_view(file, name, shape, err);
            if (!t) return false;
            *out = t->data; // 大矩阵只存指针，不拷贝
            return true;
        };
        const auto bind_vec = [&](const char *name, std::vector<uint64_t> shape,
                                  const float **out) -> bool {
            const TensorView *t = m->require_view(file, name, shape, err);
            if (!t) return false;
            *out = m->bind_f32_vector(t); // 小向量转为 fp32 副本
            return true;
        };

        // 全局权重：词嵌入、最后 norm、lm_head
        if (!bind_mat("model.embed_tokens.weight", {vocab, hidden}, &m->embed_)) return false;
        if (!bind_vec("model.norm.weight", {hidden}, &m->final_norm_)) return false;
        if (cfg.tied_embeddings) {
            // tied：lm_head 与词嵌入同源。默认共享 embed（省内存）；但若文件里
            // 带了单独量化/独立的 lm_head.weight（i4 导出选项），优先用它
            if (file.get("lm_head.weight") &&
                bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) {
                // 走 dtype 对应的正常 matvec 路径（i4 文件里即 i4 kernel）
            } else {
                m->lm_head_ = m->embed_; // 未绑定：直接共享 embed 层
                // I4 文件中 embed 存为 fp32（lookup table 不量化），lm_head 投影需走 f32 路径
                if (m->dtype_ == Dtype::kI4) m->lm_head_is_f32_ = true;
            }
        } else {
            // 非 tied embeddings：lm_head 必须独立存在
            if (!bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) return false;
        }

        // 每层权重。tensor 名字沿用 HuggingFace 约定（qwen35 的 linear_attn
        // 子模块名同样照搬），缺哪个名字就能直接定位到 exporter 的问题。
        const uint64_t gdn_v = static_cast<uint64_t>(m->gdn_value_dim_);
        const uint64_t gdn_conv = static_cast<uint64_t>(m->gdn_conv_dim_);
        const uint64_t n_v_heads = cfg.linear_num_v_heads;
        const uint64_t v_head_dim = cfg.linear_v_head_dim;
        const uint64_t conv_k = cfg.linear_conv_kernel_dim;

        m->layers_.resize(cfg.n_layers); // 预分配层权重数组
        // 逐层绑定权重
        for (uint32_t i = 0; i < cfg.n_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            LayerWeights &w = m->layers_[i];
            // norm / FFN 两种架构完全一致（SwiGLU，无 bias）
            if (!bind_vec((p + "input_layernorm.weight").c_str(), {hidden}, &w.input_ln))
                return false;
            if (!bind_vec((p + "post_attention_layernorm.weight").c_str(), {hidden}, &w.post_ln))
                return false;
            if (!bind_mat((p + "mlp.gate_proj.weight").c_str(), {inter, hidden}, &w.gate))
                return false;
            if (!bind_mat((p + "mlp.up_proj.weight").c_str(), {inter, hidden}, &w.up))
                return false;
            if (!bind_mat((p + "mlp.down_proj.weight").c_str(), {hidden, inter}, &w.down))
                return false;

            if (is_qwen35 && cfg.is_linear_layer(i)) {
                // ---- Gated DeltaNet 层（Qwen3.5 linear attention）----
                const std::string g = p + "linear_attn.";
                if (!bind_mat((g + "in_proj_qkv.weight").c_str(), {gdn_conv, hidden},
                              &w.gdn_in_qkv)) return false;
                if (!bind_mat((g + "in_proj_z.weight").c_str(), {gdn_v, hidden}, &w.gdn_in_z))
                    return false;
                if (!bind_mat((g + "in_proj_b.weight").c_str(), {n_v_heads, hidden},
                              &w.gdn_in_b)) return false;
                if (!bind_mat((g + "in_proj_a.weight").c_str(), {n_v_heads, hidden},
                              &w.gdn_in_a)) return false;
                if (!bind_mat((g + "out_proj.weight").c_str(), {hidden, gdn_v}, &w.gdn_out_proj))
                    return false;
                // conv1d 权重、A_log、dt_bias、norm 都是小向量，恒用 fp32
                if (!bind_vec((g + "conv1d.weight").c_str(), {gdn_conv, conv_k}, &w.gdn_conv_w))
                    return false;
                if (!bind_vec((g + "A_log").c_str(), {n_v_heads}, &w.gdn_a_log)) return false;
                if (!bind_vec((g + "dt_bias").c_str(), {n_v_heads}, &w.gdn_dt_bias)) return false;
                if (!bind_vec((g + "norm.weight").c_str(), {v_head_dim}, &w.gdn_norm))
                    return false;
            } else if (is_qwen35) {
                // ---- Qwen3.5 full attention 层：无 bias，q_proj 携带输出门 ----
                const std::string s = p + "self_attn.";
                // q_proj 形状 [2*q_dim, hidden]：前半是 query，后半是输出门
                if (!bind_mat((s + "q_proj.weight").c_str(), {2 * qd, hidden}, &w.q_proj))
                    return false;
                if (!bind_mat((s + "k_proj.weight").c_str(), {kvd, hidden}, &w.k_proj))
                    return false;
                if (!bind_mat((s + "v_proj.weight").c_str(), {kvd, hidden}, &w.v_proj))
                    return false;
                if (!bind_mat((s + "o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                    return false;
                // QK-Norm 权重：每头 head_dim 维
                if (!bind_vec((s + "q_norm.weight").c_str(), {cfg.head_dim}, &w.q_norm))
                    return false;
                if (!bind_vec((s + "k_norm.weight").c_str(), {cfg.head_dim}, &w.k_norm))
                    return false;
            } else {
                // ---- Qwen2.x 层：q/k/v 带 bias ----
                if (!bind_mat((p + "self_attn.q_proj.weight").c_str(), {qd, hidden}, &w.q_proj))
                    return false;
                if (!bind_mat((p + "self_attn.k_proj.weight").c_str(), {kvd, hidden}, &w.k_proj))
                    return false;
                if (!bind_mat((p + "self_attn.v_proj.weight").c_str(), {kvd, hidden}, &w.v_proj))
                    return false;
                // Qwen2.x 的 q/k/v 有 bias 向量
                if (!bind_vec((p + "self_attn.q_proj.bias").c_str(), {qd}, &w.q_bias))
                    return false;
                if (!bind_vec((p + "self_attn.k_proj.bias").c_str(), {kvd}, &w.k_bias))
                    return false;
                if (!bind_vec((p + "self_attn.v_proj.bias").c_str(), {kvd}, &w.v_bias))
                    return false;
                if (!bind_mat((p + "self_attn.o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                    return false;
            }
        }

        // KV cache 只给 full attention 层（qwen35：n_layers / interval 层）
        m->kv_.init(cfg.n_full_layers(), static_cast<int>(cfg.n_kv_heads), max_seq_len,
                    static_cast<int>(cfg.head_dim));
        // GDN 状态：Qwen3.5 的 linear attention 层需要
        if (is_qwen35) {
            const int n_linear = static_cast<int>(cfg.n_layers) - cfg.n_full_layers();
            m->gdn_state_.init(n_linear, static_cast<int>(cfg.linear_num_v_heads),
                               static_cast<int>(cfg.linear_qk_head_dim),
                               static_cast<int>(cfg.linear_v_head_dim), m->gdn_conv_dim_,
                               static_cast<int>(cfg.linear_conv_kernel_dim));
        }

        // 一次性开好所有 workspace，forward 里不再分配
        m->hidden_.resize(hidden);   // 残差流
        m->normed_.resize(hidden);   // RMSNorm 输出
        m->q_.resize(m->q_dim_);     // Query 投影
        m->k_.resize(m->kv_dim_);    // Key 投影
        m->v_.resize(m->kv_dim_);    // Value 投影
        m->attn_.resize(m->q_dim_);  // Attention 输出
        m->o_.resize(hidden);        // o_proj 输出
        m->gate_.resize(inter);      // FFN gate 支路
        m->up_.resize(inter);        // FFN up 支路
        m->ffn_.resize(hidden);      // FFN 输出
        m->logits_.resize(vocab);    // 最终 logits
        // Qwen3.5 额外 workspace
        if (is_qwen35) {
            m->q_full_.resize(2 * static_cast<size_t>(m->q_dim_)); // q_proj 全输出 [q|gate]
            m->q_gate_.resize(m->q_dim_);    // 输出门
            m->mixed_.resize(m->gdn_conv_dim_); // GDN 混合 qkv 投影
            m->z_.resize(m->gdn_value_dim_);    // GDN 门控 z
            m->b_.resize(cfg.linear_num_v_heads); // GDN 标量 b
            m->a_.resize(cfg.linear_num_v_heads); // GDN 标量 a
            m->gdn_out_.resize(m->gdn_value_dim_); // GDN 输出
        }

        *out = std::move(m); // 所有权转移
        return true;
    }

    // =========================================================================
    // QwenModel::reset() — 重置模型状态
    // =========================================================================
    // 说明：清空 KV cache 和 GDN 状态，重置 token 计数器。
    //       每条新 prompt 开始时调用。
    void QwenModel::reset() {
        kv_.reset(); // 清空 KV cache（seq_len 归零）
        if (gdn_state_.initialized()) gdn_state_.reset(); // 清空 GDN 状态
        token_count_ = 0; // 重置 token 计数
    }

    // =========================================================================
    // QwenModel::mm() — 批量矩阵乘法（GEMM，用于 prefill）
    // =========================================================================
    // 参数：
    //   w — 权重矩阵 [M x K]
    //   x — 输入矩阵 [K x N]（列主序存储，每列一个 token）
    //   y — 输出矩阵 [M x N]
    //   M — 输出维度（权重行数）
    //   K — 输入维度（权重列数）
    //   N — 批量大小（token 数量）
    // 说明：通过 backend 接口调用具体实现。用于 prefill 阶段的批量线性投影。
    void QwenModel::mm(const void *w, const float *x, float *y, int M, int K, int N) const {
        WeightTensor wt{w, quant_type_of(dtype_), M, K, group_size_};
        backend_->matmul(wt, x, y, M, K, N);
    }

} // namespace tinyqwen