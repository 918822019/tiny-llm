#pragma once

// ============================================================================
// 文件: backend.h
// 作用: 后端抽象接口定义 —— QwenModel 只调用这个接口，不关心 dtype、量化、硬件
//
// 设计理念:
//   QwenModel 通过 IBackend 接口操作所有计算，不直接依赖任何具体硬件实现。
//   每个后端（CPU/CUDA/Metal/...）实现自己的 IBackend 子类，注册到工厂函数。
//   这样 QwenModel 的代码完全与硬件无关，切换后端只需替换实现对象。
//
// 接口覆盖的计算:
//   - matvec 系列: 矩阵-向量乘法（decode 阶段的核心操作）
//   - matmul (GEMM): 矩阵-矩阵乘法（prefill 批量处理）
//   - 非 matvec 算子: RMSNorm、RoPE、attention、SwiGLU、argmax、top-k
//   - GDN 专用算子: causal conv1d、l2norm、gdn_step、门控 RMSNorm
//
// 扩展指南:
//   添加新后端时，继承 IBackend 并实现所有纯虚函数即可。
// ============================================================================

#include <cstdint>

namespace tinyqwen {

    // -------------------------------------------------------------------------
    // QuantType: 量化类型枚举
    //
    // 与文件头 dtype 字段对应，描述权重数据的存储格式。
    // 后端实现时根据此枚举选择对应的解量化/计算路径。
    // -------------------------------------------------------------------------
    enum class QuantType {
        kF32 = 0,   // FP32 权重（32位浮点，无量化）
        kF16 = 1,   // FP16 权重（16位半精度浮点，weight-only）
        kI4 = 2,    // INT4 权重量化（非对称 uint4，per-group scale+zero，HQQ interleaved）
        kVQ2 = 3,   // 2-bit 向量量化（每权重 uint8 码本索引 + per-tensor fp16 码本）
        kGPTQ = 4,  // 原生 GPTQ-INT4（AutoGPTQ 列主序 int32 打包 + 分离 scales/qzeros/g_idx，
                    //  in-band 存于 data；见 tiny_format.h 的 GptqBlockOffsets）
        // 未来扩展: kI8, kAWQ, ...
    };

    // -------------------------------------------------------------------------
    // WeightTensor: 权重张量的抽象描述
    //
    // QwenModel 不关心权重的实际 dtype，只传这个结构体给后端。
    // 后端根据 quant_type 决定如何解析 data 指针指向的内存。
    //
    // 量化参数说明:
    //   - 对于 INT4/INT8，scale 和 zero_point 已经打包在 data 里
    //     （按 interleaved 布局: 每组 [scale(2B) | zero(2B) | packed_data]）
    //   - 后端自己按 group_size 解析这些参数
    //   - 对于 FP32/FP16，data 直接就是权重值
    // -------------------------------------------------------------------------
    struct WeightTensor {
        const void* data;          // 原始数据指针（fp32/fp16/int4 等格式）
        QuantType quant_type;      // 量化类型，决定如何解读 data
        int rows;                  // 输出维度（矩阵的行数）
        int cols;                  // 输入维度（矩阵的列数）
        int group_size;            // 量化 group size（INT4/INT8 时 > 0，FP32/FP16 时为 0）
        // 注意: scale/zero 已打包在 data 里，后端自己解析，不在此处暴露
    };

    // -------------------------------------------------------------------------
    // IBackend: 计算后端抽象接口
    //
    // 所有算子的纯虚接口，每个后端实现对应一套。
    // 以下按功能分组:
    //   - matvec 系列: 矩阵乘向量（decode 阶段）
    //   - GEMM 系列: 矩阵乘矩阵（prefill 阶段）
    //   - 非 matvec 算子: 激活函数、归一化、位置编码等
    //   - GDN 专用算子: Qwen3.5 线性注意力所需
    // -------------------------------------------------------------------------
    class IBackend {
    public:
        virtual ~IBackend() = default;

        // =====================================================================
        // matvec 系列: 矩阵-向量乘法
        // decode 阶段每个 token 的核心操作，计算量最大
        // =====================================================================

        // ---------------------------------------------------------------------
        // matvec: 单路矩阵-向量乘法
        //
        // 计算: y = W @ x，其中 W 是 [out_dim, in_dim] 矩阵（行主序），x 是 [in_dim] 向量
        //
        // 参数:
        //   w:       权重矩阵（WeightTensor，可能经过量化）
        //   x:       输入向量（恒为 fp32，因为激活值保持全精度）
        //   y:       输出向量（恒为 fp32），调用方已分配 out_dim 个元素
        //   out_dim: 输出维度（矩阵行数）
        //   in_dim:  输入维度（矩阵列数）
        // ---------------------------------------------------------------------
        virtual void matvec(const WeightTensor& w, const float* x, float* y,
                           int out_dim, int in_dim) = 0;

        // ---------------------------------------------------------------------
        // matvec_pair: 成对矩阵-向量乘法
        //
        // 计算: y1 = W1 @ x, y2 = W2 @ x（共享同一个输入向量 x）
        //
        // 参数:
        //   w1, w2:  两个权重矩阵
        //   x:       输入向量（两个乘法共享）
        //   y1, y2:  输出向量
        //   out_dim: 输出维度（W1 和 W2 通常相同，如 k_proj 和 v_proj）
        //   in_dim:  输入维度
        //
        // 说明:
        //   Qwen 的 k_proj/v_proj 正是这个形状: 两个小矩阵共享输入。
        //   合并成一次调用可以摊薄同步开销，有机会走并行路径。
        // ---------------------------------------------------------------------
        virtual void matvec_pair(const WeightTensor& w1, const WeightTensor& w2,
                                 const float* x, float* y1, float* y2,
                                 int out_dim, int in_dim) = 0;

        // ---------------------------------------------------------------------
        // matvec_qkv: QKV 三路融合矩阵-向量乘法
        //
        // 计算: yq = Wq @ x, yk = Wk @ x, yv = Wv @ x
        //
        // 参数:
        //   wq, wk, wv:  query/key/value 的权重矩阵
        //   x:           输入向量
        //   yq, yk, yv:  输出向量
        //   q_dim:       query 输出维度（= n_heads * head_dim）
        //   kv_dim:      key/value 输出维度（= n_kv_heads * head_dim，GQA 时 < q_dim）
        //   in_dim:      输入维度
        //
        // 说明:
        //   q_dim 和 kv_dim 可以不同（Qwen: 896 vs 128），支持 GQA。
        //   三路融合比单独调用三次 matvec 有更好的 cache 局部性。
        // ---------------------------------------------------------------------
        virtual void matvec_qkv(const WeightTensor& wq, const WeightTensor& wk,
                                const WeightTensor& wv, const float* x,
                                float* yq, float* yk, float* yv,
                                int q_dim, int kv_dim, int in_dim) = 0;

        // =====================================================================
        // GEMM 系列: 矩阵-矩阵乘法（prefill 阶段批量处理）
        // =====================================================================

        // ---------------------------------------------------------------------
        // matmul: 矩阵-矩阵乘法
        //
        // 计算: Y[M, N] = W[M, K] @ X[K, N]
        //
        // 参数:
        //   w: 权重矩阵 [M, K] 行主序
        //   x: 输入矩阵 [K, N] 列主序（每列是一个 token 的向量）
        //   y: 输出矩阵 [M, N] 列主序
        //   M: 输出维度
        //   K: 输入维度
        //   N: token 数量（batch 维度）
        //
        // 说明:
        //   prefill 阶段一次处理 N 个 prompt token，GEMM 比逐 token matvec
        //   高效得多。N=1 时退化为 matvec，但专门的 matvec kernel 通常更快。
        // ---------------------------------------------------------------------
        virtual void matmul(const WeightTensor& w, const float* x, float* y,
                           int M, int K, int N) = 0;

        // =====================================================================
        // 非 matvec 算子: 全部走 fp32 精度
        // =====================================================================

        // ---------------------------------------------------------------------
        // rmsnorm: 均方根归一化
        //
        // 计算: y = x / sqrt(mean(x^2) + eps) * weight
        //
        // 参数:
        //   x:      输入向量 [n]
        //   weight: 可学习的缩放权重 [n]（gamma 参数）
        //   y:      输出向量 [n]
        //   n:      向量长度
        //   eps:    防止除零的小常数
        // ---------------------------------------------------------------------
        virtual void rmsnorm(const float* x, const float* weight, float* y,
                            int n, float eps) = 0;

        // ---------------------------------------------------------------------
        // rope: 旋转位置编码（RoPE），全维度旋转
        //
        // 参数:
        //   q, k:        query 和 key 向量，就地修改
        //   n_heads:     query 注意力头数
        //   n_kv_heads:  key/value 注意力头数
        //   head_dim:    每个头的维度
        //   pos:         当前 token 在序列中的位置
        //   theta:       RoPE 的底数（如 10000.0）
        //
        // 说明:
        //   与 HF Qwen2 的 rotate-half 约定一致:
        //     out[i]        = x[i] * cos - x[i + half] * sin
        //     out[i + half] = x[i + half] * cos + x[i] * sin
        // ---------------------------------------------------------------------
        virtual void rope(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int pos, float theta) = 0;

        // ---------------------------------------------------------------------
        // partial_rope: 部分旋转位置编码
        //
        // 参数:
        //   同 rope，但只旋转每个 head 的前 rotary_dim 个分量
        //
        // 说明:
        //   Qwen3.5 使用 partial RoPE：只旋转 head_dim 的一部分（如 25%），
        //   其余维度保持不变。rotary_dim = head_dim * partial_rotary_factor。
        // ---------------------------------------------------------------------
        virtual void partial_rope(float* q, float* k, int n_heads, int n_kv_heads,
                                 int head_dim, int rotary_dim, int pos, float theta) = 0;

        // ---------------------------------------------------------------------
        // attention_decode: decode 阶段的注意力计算
        //
        // 计算: 对单个 query 位置，与 KV cache 中所有历史位置做 attention
        //
        // 参数:
        //   q:            当前 query 向量 [n_heads * head_dim]
        //   k_cache:      key 缓存 [n_kv_heads][max_seq_len][head_dim]
        //   v_cache:      value 缓存 [n_kv_heads][max_seq_len][head_dim]
        //   seq_len:      缓存中有效位置数 [0, seq_len)
        //   max_seq_len:  KV 缓存容量上限
        //   n_heads:      query 头数
        //   n_kv_heads:   key/value 头数（GQA 时 < n_heads）
        //   head_dim:     每个头的维度
        //   scale:        attention 缩放因子（= 1/sqrt(head_dim)）
        //   out:          输出向量 [n_heads * head_dim]
        //
        // 说明:
        //   GQA 映射: query head h 对应的 kv head 为 h / (n_heads / n_kv_heads)。
        //   采用 online softmax，不需要暂存整个 score 矩阵。
        // ---------------------------------------------------------------------
        virtual void attention_decode(const float* q, const float* k_cache,
                                     const float* v_cache, int seq_len, int max_seq_len,
                                     int n_heads, int n_kv_heads, int head_dim,
                                     float scale, float* out) = 0;

        // ---------------------------------------------------------------------
        // attention_decode_f16kv: fp16-KV 融合 attention
        //
        // 与 attention_decode 相同的数学，但 k_cache/v_cache 为 fp16（uint16_t*），
        // 在寄存器内转 fp32 计算。消灭独立反量化遍历。仅 fp16-KV cache 时调用。
        // ---------------------------------------------------------------------
        virtual void attention_decode_f16kv(const float* q, const uint16_t* k_cache,
                                           const uint16_t* v_cache, int seq_len, int max_seq_len,
                                           int n_heads, int n_kv_heads, int head_dim,
                                           float scale, float* out) = 0;

        // ---------------------------------------------------------------------
        // swiglu: SwiGLU 激活函数
        //
        // 计算: gate[i] = silu(gate[i]) * up[i]（就地修改 gate）
        //
        // 参数:
        //   gate: 门控值向量 [n]，就地修改为 silu(gate) * up
        //   up:   上投影值向量 [n]
        //   n:    向量长度
        //
        // 说明:
        //   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        //   SwiGLU 是 Qwen FFN 的核心激活函数。
        // ---------------------------------------------------------------------
        virtual void swiglu(float* gate, const float* up, int n) = 0;

        // ---------------------------------------------------------------------
        // argmax: 返回数组中最大值的下标
        //
        // 参数:
        //   logits: 输入数组 [n]（通常是 logits 向量）
        //   n:      数组长度
        //
        // 返回值:
        //   第一个最大值的下标（平局取靠前者）
        //
        // 说明:
        //   greedy decoding 时用此函数选取概率最高的 token。
        // ---------------------------------------------------------------------
        virtual int argmax(const float* logits, int n) = 0;

        // ---------------------------------------------------------------------
        // top_k_logits: 取 logits 中 top-k 最大的值和下标
        //
        // 参数:
        //   logits:  输入数组 [n]
        //   n:       数组长度
        //   k:       选取前 k 个最大值
        //   indices: 输出参数，top-k 的下标 [k]
        //   values:  输出参数，top-k 的值 [k]
        //
        // 说明:
        //   用于调试和展示 top-k 候选 token 及其概率。
        // ---------------------------------------------------------------------
        virtual void top_k_logits(const float* logits, int n, int k,
                                 int* indices, float* values) = 0;

        // ---------------------------------------------------------------------
        // topk_softmax: MoE 路由门用——取 logits 前 k 大、做 softmax 归一
        //
        // 计算: 从 gate_logits[n] 选 k 个最大值，记其下标到 indices[k]，
        //       并对这 k 个 logits 做 softmax（减最大值稳定）写入 weights[k]，
        //       weights 和为 1。indices 按分数降序。
        //
        // 参数:
        //   gate_logits: 路由门输出 [n_experts]
        //   n:           专家数
        //   k:           激活专家数（top-k）
        //   indices:     输出，选中的专家下标 [k]
        //   weights:     输出，归一化后的路由权重 [k]
        // ---------------------------------------------------------------------
        virtual void topk_softmax(const float* gate_logits, int n, int k,
                                   int* indices, float* weights) = 0;

        // =====================================================================
        // GDN 专用算子（Qwen3.5 线性注意力层需要）
        // =====================================================================

        // ---------------------------------------------------------------------
        // causal_conv1d_update: 因果深度卷积的单步更新
        //
        // 参数:
        //   x:           当前输入 [dim]
        //   state:       卷积状态 [dim][kernel_size-1]，保存历史输入，就地更新
        //   weight:      卷积权重 [dim][kernel_size]
        //   out:         输出 [dim]
        //   dim:         通道数
        //   kernel_size: 卷积核大小
        //
        // 说明:
        //   out[c] = silu( sum_i w[c][i] * [state_c..., x[c]][i] )
        //   同时把 x[c] 推入 state（最旧的值被挤出）。
        // ---------------------------------------------------------------------
        virtual void causal_conv1d_update(float* x, float* state, const float* weight,
                                         float* out, int dim, int kernel_size) = 0;

        // ---------------------------------------------------------------------
        // l2norm_inplace: L2 归一化（就地）
        //
        // 计算: x = x / sqrt(sum(x^2) + eps)
        //
        // 参数:
        //   x:   输入向量 [n]，就地修改
        //   n:   向量长度
        //   eps: 防止除零的小常数
        // ---------------------------------------------------------------------
        virtual void l2norm_inplace(float* x, int n, float eps) = 0;

        // ---------------------------------------------------------------------
        // gdn_step: Gated DeltaNet 的单个 head 递归步
        //
        // 参数:
        //   S:      状态矩阵 [qk_dim, v_dim]，行主序，就地更新
        //   q, k:   已 l2norm 的 query/key 向量 [qk_dim]
        //   v:      value 向量 [v_dim]
        //   g:      衰减对数（负数），kernel 内部取 exp
        //   beta:   写入门（已 sigmoid）
        //   o:      输出向量 [qk_dim]
        //   qk_dim: Q/K 头的维度
        //   v_dim:  V 头的维度
        //
        // 说明:
        //   S = exp(g) * S + outer(k, beta * (v - S^T @ k))
        //   o = S^T @ q
        // ---------------------------------------------------------------------
        virtual void gdn_step(float* S, const float* q, const float* k, const float* v,
                             float g, float beta, float* o, int qk_dim, int v_dim) = 0;

        // ---------------------------------------------------------------------
        // rmsnorm_gated: 带门控的 RMSNorm
        //
        // 计算: y = (x / sqrt(mean(x^2) + eps) * weight) * silu(gate)
        //
        // 参数:
        //   x:      输入向量 [n]
        //   gate:   门控向量 [n]（用于 silu 激活）
        //   weight: 可学习的缩放权重 [n]
        //   y:      输出向量 [n]
        //   n:      向量长度
        //   eps:    防止除零的小常数
        //
        // 说明:
        //   GDN 输出在 out_proj 之前按 v head 维度做此归一化。
        // ---------------------------------------------------------------------
        virtual void rmsnorm_gated(float* x, const float* gate, const float* weight,
                                  float* y, int n, float eps) = 0;
    };

} // namespace tinyqwen