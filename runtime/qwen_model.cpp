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
                case Dtype::kVQ2: return QuantType::kVQ2;
                case Dtype::kGPTQ4: return QuantType::kGPTQ;
                default:          return QuantType::kF32;
            }
        }

        // find_hadamard_block_size — 与 kronq/hadamard.py 一致的块大小选择：
        // 取能整除 dim 的最大候选（256/128/64/32，其次 16/8/4/2，兜底 1）。
        int find_hadamard_block_size(int dim) {
            const int cand[] = {256, 128, 64, 32, 16, 8, 4, 2};
            for (int bs : cand) {
                if (dim % bs == 0) return bs;
            }
            return 1;
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
        // dtype 校验：量化文件（i4/vq2/gptq）与 MoE 文件允许混合 dtype
        // （大矩阵量化、小向量/embed kF32/kF16）。MoE 文件 master 可为 kF32 而
        // 路由专家为 kGPTQ4，故 MoE 额外放行 kGPTQ4。
        const bool mixed_ok = [&] {
            if (t->dtype == dtype_) return true;                        // 与主 dtype 一致
            if (t->dtype == Dtype::kF32) return true;                   // 小向量/embed 恒可 f32
            if (dtype_ == Dtype::kI4 && t->dtype == Dtype::kI4) return true;
            if (dtype_ == Dtype::kVQ2 && (t->dtype == Dtype::kVQ2 ||
                                          t->dtype == Dtype::kF16 ||
                                          t->dtype == Dtype::kI4)) return true;
            if (cfg_.is_moe() && t->dtype == Dtype::kGPTQ4) return true;
            if (dtype_ == Dtype::kGPTQ4 && t->dtype == Dtype::kGPTQ4) return true;
            // GPTQ / MoE 文件放行 fp16：真 checkpoint 的 norm / embed / lm_head
            // 本来就是 fp16，保留源 dtype 而非升 fp32（升了只是白白翻倍，实测无损）。
            if ((dtype_ == Dtype::kGPTQ4 || cfg_.is_moe()) &&
                t->dtype == Dtype::kF16) return true;
            return false;
        }();
        if (!mixed_ok) {
            if (err)
                *err = "tensor " + name + " dtype mismatch: got " + dtype_name(t->dtype) +
                       " expected " + dtype_name(dtype_) + " (or allowed mixed)";
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
        WeightTensor wt{w, quant_type_of(dtype_), out_dim, in_dim, cur_group_size()};
        backend_->matvec(wt, x, y, out_dim, in_dim);
    }

    void QwenModel::mv_typed(const void *w, const float *x, float *y, int out_dim, int in_dim,
                             Dtype d, int group_size) const {
        WeightTensor wt{w, quant_type_of(d), out_dim, in_dim, group_size};
        backend_->matvec(wt, x, y, out_dim, in_dim);
    }

    // =========================================================================
    // QwenModel::mv_rot() — 旋转感知 matvec
    // =========================================================================
    // 若 rot 有效（sign 非空），先把输入 x 经 BiIP 配对旋转写入 rot_buf_，
    // 再以旋转后的激活做 matvec；否则直接 mv（非旋转模型无额外开销）。
    // 自抵消保证：旋转激活 × 旋转量化权重 == 原始激活 × 原始权重。
    void QwenModel::mv_rot(const void *w, const float *x, float *y, int out_dim, int in_dim,
                           const RotParams &rot) const {
        const float *xr = x;
        if (rot.sign) {
            biip_rotate_activation(x, rot_buf_.data(), in_dim, rot.scale, rot.sign,
                                   rot.block_size);
            xr = rot_buf_.data();
        }
        mv(w, xr, y, out_dim, in_dim);
    }

    // =========================================================================
    // QwenModel::mm_rot() — 旋转感知 GEMM（prefill）
    // =========================================================================
    // X 为列主序 [K, N]（每列一个 token）。若 rot 有效，逐列旋转进 rot_buf_ 再 GEMM。
    void QwenModel::mm_rot(const void *w, const float *x, float *y, int M, int K, int N,
                           const RotParams &rot) const {
        const float *xr = x;
        if (rot.sign) {
            const size_t total = static_cast<size_t>(K) * N;
            if (rot_buf_.size() < total) rot_buf_.resize(total);
            for (int c = 0; c < N; ++c) {
                biip_rotate_activation(x + static_cast<size_t>(c) * K,
                                       rot_buf_.data() + static_cast<size_t>(c) * K,
                                       K, rot.scale, rot.sign, rot.block_size);
            }
            xr = rot_buf_.data();
        }
        mm(w, xr, y, M, K, N);
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
        WeightTensor wt1{w1, quant_type_of(dtype_), out_dim, in_dim, cur_group_size()};
        WeightTensor wt2{w2, quant_type_of(dtype_), out_dim, in_dim, cur_group_size()};
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
        WeightTensor wqt{wq, quant_type_of(dtype_), q_dim, in_dim, cur_group_size()};
        WeightTensor wkt{wk, quant_type_of(dtype_), kv_dim, in_dim, cur_group_size()};
        WeightTensor wvt{wv, quant_type_of(dtype_), kv_dim, in_dim, cur_group_size()};
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
                           std::unique_ptr<IBackend> backend, bool kv_fp16,
                           ExpertStore *expert_store) {
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

        // MoE 配置（仅 kQwen35MoE 有意义；非 MoE 全为 0/默认）
        m->expert_store_ = expert_store;
        if (cfg.is_moe()) {
            m->moe_inter_ = static_cast<int>(cfg.moe_intermediate_size);
            m->shared_inter_ = static_cast<int>(cfg.shared_expert_intermediate_size);
            m->n_experts_ = static_cast<int>(cfg.n_routed_experts);
            m->experts_per_tok_ = static_cast<int>(cfg.num_experts_per_tok);
            m->gptq_group_size_ = static_cast<int>(cfg.gptq_group_size);
            // 专家 dtype 须文件里实际专家 tensor 的 dtype 决定（fake 模型 = kGPTQ4）
            // 在逐层绑定时从首个专家 tensor 读出，这里先按 GPTQ 预设（fake）。
            m->expert_dtype_ = (m->gptq_group_size_ > 0) ? Dtype::kGPTQ4 : Dtype::kF32;
        }

        // Qwen3.5 混合架构：预计算 GDN 和 partial RoPE 的派生维度
        // MoE 与 dense 共用同一套 attention（GDN+full 混合），故 attention 形
        // 状校验与 workspace 对两者都适用。
        const bool is_qwen35 = cfg.uses_qwen35_attention();
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
        // 可选小向量：文件里没有就留 nullptr，存在则照常校验形状。
        // Qwen2.5 有 attention bias 无 QK norm，Qwen3 稠密反之——两者共用
        // MODEL_QWEN2 族分支，靠权重存在性区分（与 bind_rot 同一惯例）。
        const auto opt_vec = [&](const char *name, std::vector<uint64_t> shape,
                                 const float **out) -> bool {
            if (!file.get(name)) return true;
            return bind_vec(name, std::move(shape), out);
        };

        // 绑定单个子层的 BiIP 旋转参数（可选）：存在 {prefix}.rot_sign 才视为被旋转。
        // sign/scale 存 f16，转 fp32 副本；block_size 由 in_f 推导。命中即置 rotated_。
        const auto bind_rot = [&](const std::string &prefix, int in_f, RotParams *rot) {
            const TensorView *sign_t = file.get(prefix + ".rot_sign");
            if (!sign_t) return;                       // 该子层未旋转
            rot->sign = m->bind_f32_vector(sign_t);
            const TensorView *scale_t = file.get(prefix + ".rot_scale");
            rot->scale = scale_t ? m->bind_f32_vector(scale_t) : nullptr;
            rot->block_size = find_hadamard_block_size(in_f);
            m->rotated_ = true;
        };

        // 全局权重：词嵌入、最后 norm、lm_head
        {
            const TensorView *ev = file.get("model.embed_tokens.weight");
            if (!ev) {
                *err = "missing tensor: model.embed_tokens.weight";
                return false;
            }
            m->embed_dtype_ = ev->dtype;  // embed 真实 dtype
            if (ev->data == nullptr) {
                // embed 已卸载到 SSD：记偏移，查表时按需 pread 单行。
                // bind_mat 会因 data==nullptr 失败，故这里单独处理。
                m->embed_ = nullptr;
                m->embed_file_offset_ = ev->file_offset;
                m->embed_row_.resize(static_cast<size_t>(hidden));
            } else if (!bind_mat("model.embed_tokens.weight", {vocab, hidden}, &m->embed_)) {
                return false;
            }
        }
        if (!bind_vec("model.norm.weight", {hidden}, &m->final_norm_)) return false;
        if (cfg.tied_embeddings) {
            // tied：lm_head 与词嵌入同源。默认共享 embed（省内存）；但若文件里
            // 带了单独量化/独立的 lm_head.weight（i4 导出选项），优先用它
            if (file.get("lm_head.weight") &&
                bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) {
                // 走 dtype 对应的正常 matvec 路径（i4 文件里即 i4 kernel）
            } else {
                m->lm_head_ = m->embed_; // 未绑定：直接共享 embed 层
                // lm_head 投影按 embed 的真实 dtype 路由（i4 embed=f32；vq2 embed 可为 f16）
                if (m->embed_dtype_ == Dtype::kF32) m->lm_head_is_f32_ = true;
                else if (m->embed_dtype_ == Dtype::kF16) m->lm_head_is_f16_ = true;
                else if (m->embed_dtype_ == Dtype::kI4) m->lm_head_is_i4_ = true;
            }
        } else {
            // 非 tied embeddings：lm_head 必须独立存在
            if (!bind_mat("lm_head.weight", {vocab, hidden}, &m->lm_head_)) return false;
            m->lm_head_dtype_ = file.get("lm_head.weight")->dtype;
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
            if (cfg.is_moe()) {
                // ---- MoE FFN：路由门 + 共享专家 + 路由专家 ----
                const uint64_t moe_inter = cfg.moe_intermediate_size;
                const uint64_t shared_inter = cfg.shared_expert_intermediate_size;
                const uint64_t n_exp = cfg.n_routed_experts;
                if (!bind_mat((p + "mlp.gate.weight").c_str(), {n_exp, hidden}, &w.moe_router))
                    return false;
                // router 常是 fp32 而 master dtype 可能是 kGPTQ4，须记自身 dtype
                if (i == 0) m->moe_router_dtype_ = file.get((p + "mlp.gate.weight").c_str())->dtype;
                // 共享专家可选：Qwen3-MoE 没有这组权重，文件里不存在这些 tensor，
                // 无条件绑定会报 "missing tensor"。
                if (cfg.has_shared_expert()) {
                    if (!bind_mat((p + "mlp.shared_experts.gate_proj.weight").c_str(),
                                  {shared_inter, hidden}, &w.moe_shared_gate)) return false;
                    if (!bind_mat((p + "mlp.shared_experts.up_proj.weight").c_str(),
                                  {shared_inter, hidden}, &w.moe_shared_up)) return false;
                    if (!bind_mat((p + "mlp.shared_experts.down_proj.weight").c_str(),
                                  {hidden, shared_inter}, &w.moe_shared_down)) return false;
                }
                // 路由专家：resident 模式绑定内存指针；稀疏加载下 data==nullptr，
                // 只注册文件 offset 给 ExpertStore 按需 pread。
                w.moe_experts.resize(n_exp);
                for (uint32_t e = 0; e < n_exp; ++e) {
                    const std::string pe = p + "mlp.experts." + std::to_string(e) + ".";
                    const TensorView *tg = m->require_view(file, pe + "gate_proj.weight",
                                                           {moe_inter, hidden}, err);
                    if (!tg) return false;
                    const TensorView *tu = m->require_view(file, pe + "up_proj.weight",
                                                           {moe_inter, hidden}, err);
                    if (!tu) return false;
                    const TensorView *td = m->require_view(file, pe + "down_proj.weight",
                                                           {hidden, moe_inter}, err);
                    if (!td) return false;
                    // 专家被卸载（留盘）时必须有 ExpertStore 兜住。放行 nullptr
                    // 进 forward 会段错误或静默算错，这里 fail fast。
                    if (tg->data == nullptr && !expert_store) {
                        if (err) *err = "expert weights are SSD-offloaded but no ExpertStore "
                                        "given; load the model with offload_experts only when "
                                        "an ExpertStore is wired up";
                        return false;
                    }
                    w.moe_experts[e] = {tg->data, tu->data, td->data};
                    m->expert_dtype_ = tg->dtype;  // 推断（fake = kGPTQ4）
                    if (expert_store) {
                        // offset 一律取 TensorView::file_offset：稀疏紧凑重排后
                        // (data - base) 不再是文件内偏移，指针算式会算错。
                        expert_store->register_expert(
                            static_cast<int>(i), static_cast<int>(e),
                            tg->file_offset, tg->nbytes,
                            tu->file_offset, tu->nbytes,
                            td->file_offset, td->nbytes,
                            static_cast<int>(moe_inter), static_cast<int>(hidden),
                            m->gptq_group_size_);
                    }
                }
            } else {
                // ---- dense SwiGLU FFN ----
                if (!bind_mat((p + "mlp.gate_proj.weight").c_str(), {inter, hidden}, &w.gate))
                    return false;
                if (!bind_mat((p + "mlp.up_proj.weight").c_str(), {inter, hidden}, &w.up))
                    return false;
                if (!bind_mat((p + "mlp.down_proj.weight").c_str(), {hidden, inter}, &w.down))
                    return false;
            }

            if (is_qwen35 && cfg.is_linear_layer(i)) {
                // ---- Gated DeltaNet 层（Qwen3.5 linear attention）----
                const std::string g = p + "linear_attn.";
                if (!bind_mat((g + "in_proj_qkv.weight").c_str(), {gdn_conv, hidden},
                              &w.gdn_in_qkv)) return false;
                // 记录 GDN 投影真实 dtype。真 checkpoint 的 GDN 投影是 bf16/fp16
                // （不是 GPTQ），而模型级 dtype_ 是 kGPTQ4。forward 必须用
                // mv_typed 按此 dtype 路由，否则 fp16 数据被当 GPTQ 解析 → 乱码。
                w.gdn_dtype = file.get((g + "in_proj_qkv.weight").c_str())->dtype;
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
                // 记录 attention 投影真实 dtype。真 checkpoint 的 attention 投影是
                // bf16/fp16（不是 GPTQ），forward 必须用 mv_typed 按此 dtype 路由。
                w.attn_dtype = file.get((s + "q_proj.weight").c_str())->dtype;
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
                // q/k/v bias：Qwen2.x 有，Qwen3 稠密（attention_bias=false）没有
                if (!opt_vec((p + "self_attn.q_proj.bias").c_str(), {qd}, &w.q_bias))
                    return false;
                if (!opt_vec((p + "self_attn.k_proj.bias").c_str(), {kvd}, &w.k_bias))
                    return false;
                if (!opt_vec((p + "self_attn.v_proj.bias").c_str(), {kvd}, &w.v_bias))
                    return false;
                // QK per-head RMSNorm：Qwen3 稠密有，Qwen2.x 没有
                if (!opt_vec((p + "self_attn.q_norm.weight").c_str(), {cfg.head_dim}, &w.q_norm))
                    return false;
                if (!opt_vec((p + "self_attn.k_norm.weight").c_str(), {cfg.head_dim}, &w.k_norm))
                    return false;
                if ((w.q_norm == nullptr) != (w.k_norm == nullptr)) {
                    if (err)
                        *err = "layer " + std::to_string(i) +
                               ": q_norm/k_norm must both be present or both absent";
                    return false;
                }
                if (!bind_mat((p + "self_attn.o_proj.weight").c_str(), {hidden, qd}, &w.o_proj))
                    return false;
                // BiIP 旋转参数（旋转量化模型才有；否则全部 no-op）
                bind_rot(p + "self_attn.q_proj", static_cast<int>(hidden), &w.rot_q);
                bind_rot(p + "self_attn.k_proj", static_cast<int>(hidden), &w.rot_k);
                bind_rot(p + "self_attn.v_proj", static_cast<int>(hidden), &w.rot_v);
                bind_rot(p + "self_attn.o_proj", static_cast<int>(qd), &w.rot_o);
                bind_rot(p + "mlp.gate_proj", static_cast<int>(hidden), &w.rot_gate);
                bind_rot(p + "mlp.up_proj", static_cast<int>(hidden), &w.rot_up);
                bind_rot(p + "mlp.down_proj", static_cast<int>(inter), &w.rot_down);
            }
        }

        // KV cache 只给 full attention 层（qwen35：n_layers / interval 层）
        m->kv_.init(cfg.n_full_layers(), static_cast<int>(cfg.n_kv_heads), max_seq_len,
                    static_cast<int>(cfg.head_dim), kv_fp16);
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
        // 旋转激活暂存：最大 in_dim（hidden / inter / q_dim 取大）
        m->rot_buf_.resize(std::max({hidden, inter, qd}));
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

        // MoE workspace（仅 kQwen35MoE）
        if (cfg.is_moe()) {
            m->moe_gate_logits_.resize(cfg.n_routed_experts);
            m->moe_expert_gate_.resize(m->moe_inter_);
            m->moe_expert_up_.resize(m->moe_inter_);
            m->moe_expert_out_.resize(hidden);
            m->moe_ffn_acc_.resize(hidden);
            m->moe_shared_out_.resize(hidden);
            m->moe_topk_idx_.resize(m->experts_per_tok_);
            m->moe_topk_w_.resize(m->experts_per_tok_);
            // dense FFN workspace 仍按 inter 分配（非 MoE 层/兜底用；MoE 不用它）
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
        WeightTensor wt{w, quant_type_of(dtype_), M, K, cur_group_size()};
        backend_->matmul(wt, x, y, M, K, N);
    }

    // =========================================================================
    // QwenModel::attention_kv() — attention 统一分发（按 KV cache 精度）
    // =========================================================================
    // fp32 KV：走标准 attention_decode（读 fp32）。
    // fp16 KV：走融合 attention_decode_f16kv（读 fp16、寄存器内转 fp32），
    //   消灭独立反量化遍历。
    void QwenModel::attention_kv(const float *q, int layer, int seq, int n_heads,
                                 int n_kv_heads, int head_dim, float scale, float *out) const {
        if (kv_.use_fp16()) {
            backend_->attention_decode_f16kv(q, kv_.k_f16(layer), kv_.v_f16(layer), seq,
                                             max_seq_len_, n_heads, n_kv_heads, head_dim, scale,
                                             out);
        } else {
            backend_->attention_decode(q, kv_.k(layer), kv_.v(layer), seq, max_seq_len_, n_heads,
                                       n_kv_heads, head_dim, scale, out);
        }
    }

} // namespace tinyqwen