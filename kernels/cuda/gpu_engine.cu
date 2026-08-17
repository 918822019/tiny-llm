// GPU-resident decode engine：整段 forward 常驻显存。
//
// ============================================================================
// 为什么：消灭"每 matvec 一次 CPU↔GPU 桥接"
// ============================================================================
// CPU 驱动的 forward（runtime/qwen_model.cpp::forward_token）里，每次 matvec
// 都要 H2D 传 x、算完 D2H 传 y 回 CPU，非 matvec 算子也在 CPU 跑——这些桥接
// + 同步是当前离带宽地板（~1.7ms）还差 ~4ms 的主因。本 engine 把权重、激活、
// KV cache 全部常驻显存，decode_step 在单条 stream 上串起整个前向，每步只有
// 最后一次 argmax 结果（4 字节）D2H。CPU forward 原样保留作参考。
//
// 经 dispatch 的 decode-engine registry 自注册（名字 "cuda"）；main.cpp 用
// --engine cuda 选用。无 CUDA 构建时本文件不编译、registry 为空、自动回退 CPU。
//
// 数值口径：与 forward_token 逐 op 对齐（matvec 用 coalesced fp32 累加、
// attention online softmax + double 点积、rmsnorm double 平方和、rope/swiglu
// 与 ref 同精度函数）。greedy argmax 与 CPU 平局规则一致（严格 > 取首个）。

#include "dispatch.h"
#include "ref_ops.h" // half_to_float（f16 norm/bias 转 fp32）

#include "cuda/gpu_kernels.cuh"

#include "model_loader.h" // ModelFile / ModelConfig / TensorView
#include "tiny_format.h"  // Dtype

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace tinyqwen {
    namespace {
        inline void echeck(cudaError_t err, const char *what) {
            if (err != cudaSuccess) {
                std::fprintf(stderr, "tinyqwen gpu engine: CUDA 失败于 %s：%s\n", what,
                             cudaGetErrorString(err));
                std::abort();
            }
        }
#define ENG_CHECK(expr) echeck((expr), #expr)
    } // namespace

    // engine 的完整 device 状态。dispatch.h 里只有前向声明（opaque），这里定义。
    struct GpuDecodeEngine {
        // 模型配置
        int n_layers = 0, hidden = 0, inter = 0, vocab = 0;
        int n_heads = 0, n_kv_heads = 0, head_dim = 0, max_seq_len = 0;
        int q_dim = 0, kv_dim = 0;
        float attn_scale = 0, rms_eps = 0, rope_theta = 0;
        bool is_f16 = false;
        cudaStream_t stream = nullptr;

        // device 权重（大矩阵按模型 dtype 原样常驻；norm/bias 恒 fp32）
        void *d_embed = nullptr;   // [vocab,hidden]
        void *d_lm_head = nullptr; // [vocab,hidden]（tied 时 = d_embed）
        float *d_final_norm = nullptr; // [hidden] fp32
        struct LayerDev {
            float *input_ln = nullptr, *post_ln = nullptr; // fp32 [hidden]
            float *qb = nullptr, *kb = nullptr, *vb = nullptr; // fp32 bias
            void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr; // dtype 矩阵
            void *gate = nullptr, *up = nullptr, *down = nullptr;       // dtype 矩阵
        };
        std::vector<LayerDev> layers;

        // 激活（fp32）
        float *d_hidden = nullptr, *d_normed = nullptr, *d_q = nullptr, *d_k = nullptr,
              *d_v = nullptr, *d_attn = nullptr, *d_o = nullptr, *d_gate = nullptr,
              *d_up = nullptr, *d_ffn = nullptr, *d_logits = nullptr;

        // KV cache：与 host KvCache 同款 arena（K 前半、V 后半），fp32。
        float *d_kv = nullptr;
        size_t layer_stride = 0; // n_kv_heads*max_seq_len*head_dim

        // argmax 结果（device），seq_len 由 host 计数
        int *d_next = nullptr;
        int seq_len = 0;
    };

    namespace {
        // 原样上传一个 tensor（保持 dtype）。
        bool upload_raw(const ModelFile *file, const char *name, void **dst, std::string *err) {
            const TensorView *t = file->get(name);
            if (!t) {
                if (err) *err = std::string("gpu engine: 缺少 tensor ") + name;
                return false;
            }
            ENG_CHECK(cudaMalloc(dst, t->nbytes));
            ENG_CHECK(cudaMemcpy(*dst, t->data, t->nbytes, cudaMemcpyHostToDevice));
            return true;
        }
        // 上传小向量为 fp32（f16 模型先在 host 转 fp32，对齐 bind_f32_vector）。
        bool upload_vec_f32(const ModelFile *file, const char *name, bool is_f16, float **dst,
                            std::string *err) {
            const TensorView *t = file->get(name);
            if (!t) {
                if (err) *err = std::string("gpu engine: 缺少 tensor ") + name;
                return false;
            }
            const size_t n = t->numel();
            std::vector<float> host(n);
            if (is_f16) {
                const uint8_t *src = t->data;
                for (size_t i = 0; i < n; ++i) {
                    uint16_t h;
                    std::memcpy(&h, src + i * 2, sizeof(h));
                    host[i] = half_to_float(h);
                }
            } else {
                std::memcpy(host.data(), t->data, n * sizeof(float));
            }
            ENG_CHECK(cudaMalloc(reinterpret_cast<void **>(dst), n * sizeof(float)));
            ENG_CHECK(cudaMemcpy(*dst, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
            return true;
        }

        // dtype 分派的 matvec 包装（engine 内部用）。
        inline void mv(const GpuDecodeEngine *e, const void *w, const float *x, float *y,
                       int out_dim, int in_dim) {
            if (e->is_f16)
                gpu::matvec_f16(e->stream, reinterpret_cast<const uint16_t *>(w), x, y, out_dim,
                                in_dim);
            else
                gpu::matvec_f32(e->stream, reinterpret_cast<const float *>(w), x, y, out_dim,
                                in_dim);
        }
        inline void mv_pair(const GpuDecodeEngine *e, const void *w1, const void *w2,
                            const float *x, float *y1, float *y2, int out_dim, int in_dim) {
            if (e->is_f16)
                gpu::matvec_f16_pair(e->stream, reinterpret_cast<const uint16_t *>(w1),
                                     reinterpret_cast<const uint16_t *>(w2), x, y1, y2, out_dim,
                                     in_dim);
            else
                gpu::matvec_f32_pair(e->stream, reinterpret_cast<const float *>(w1),
                                     reinterpret_cast<const float *>(w2), x, y1, y2, out_dim,
                                     in_dim);
        }
        inline void mv_qkv(const GpuDecodeEngine *e, const void *wq, const void *wk,
                           const void *wv, const float *x, float *yq, float *yk, float *yv,
                           int q_dim, int kv_dim, int in_dim) {
            if (e->is_f16)
                gpu::matvec_f16_qkv(e->stream, reinterpret_cast<const uint16_t *>(wq),
                                    reinterpret_cast<const uint16_t *>(wk),
                                    reinterpret_cast<const uint16_t *>(wv), x, yq, yk, yv, q_dim,
                                    kv_dim, in_dim);
            else
                gpu::matvec_f32_qkv(e->stream, reinterpret_cast<const float *>(wq),
                                    reinterpret_cast<const float *>(wk),
                                    reinterpret_cast<const float *>(wv), x, yq, yk, yv, q_dim,
                                    kv_dim, in_dim);
        }
    } // namespace

    // ========================================================================
    // create：解析配置、上传全部权重、开好激活/KV/argmax 缓冲。
    // ========================================================================
    static bool engine_create(const void *model_file, int max_seq_len, std::string *err,
                              GpuDecodeEngine **out) {
        const ModelFile *file = static_cast<const ModelFile *>(model_file);
        if (!file || !file->loaded()) {
            if (err) *err = "gpu engine: model file 未加载";
            return false;
        }
        const ModelConfig &cfg = file->config();
        if (max_seq_len <= 0) max_seq_len = static_cast<int>(cfg.max_seq_len);
        if (max_seq_len > static_cast<int>(cfg.max_seq_len)) {
            if (err)
                *err = "gpu engine: max_seq_len " + std::to_string(max_seq_len) + " 超过模型上限 " +
                       std::to_string(cfg.max_seq_len);
            return false;
        }

        GpuDecodeEngine *e = new GpuDecodeEngine();
        e->n_layers = static_cast<int>(cfg.n_layers);
        e->hidden = static_cast<int>(cfg.hidden_size);
        e->inter = static_cast<int>(cfg.intermediate_size);
        e->vocab = static_cast<int>(cfg.vocab_size);
        e->n_heads = static_cast<int>(cfg.n_heads);
        e->n_kv_heads = static_cast<int>(cfg.n_kv_heads);
        e->head_dim = static_cast<int>(cfg.head_dim);
        e->max_seq_len = max_seq_len;
        e->q_dim = e->n_heads * e->head_dim;
        e->kv_dim = e->n_kv_heads * e->head_dim;
        e->attn_scale = 1.0f / std::sqrt(static_cast<float>(e->head_dim));
        e->rms_eps = cfg.rms_norm_eps;
        e->rope_theta = cfg.rope_theta;
        e->is_f16 = (file->header().dtype == static_cast<uint32_t>(Dtype::kF16));

        ENG_CHECK(cudaStreamCreate(&e->stream));

        // 全局权重
        if (!upload_raw(file, "model.embed_tokens.weight", &e->d_embed, err)) return false;
        if (!upload_vec_f32(file, "model.norm.weight", e->is_f16, &e->d_final_norm, err))
            return false;
        if (cfg.tied_embeddings) {
            e->d_lm_head = e->d_embed; // tied：共享
        } else {
            if (!upload_raw(file, "lm_head.weight", &e->d_lm_head, err)) return false;
        }

        // 每层权重
        e->layers.resize(e->n_layers);
        for (int i = 0; i < e->n_layers; ++i) {
            const std::string p = "model.layers." + std::to_string(i) + ".";
            auto &L = e->layers[i];
            if (!upload_vec_f32(file, (p + "input_layernorm.weight").c_str(), e->is_f16,
                                &L.input_ln, err)) return false;
            if (!upload_vec_f32(file, (p + "post_attention_layernorm.weight").c_str(), e->is_f16,
                                &L.post_ln, err)) return false;
            if (!upload_vec_f32(file, (p + "self_attn.q_proj.bias").c_str(), e->is_f16, &L.qb, err))
                return false;
            if (!upload_vec_f32(file, (p + "self_attn.k_proj.bias").c_str(), e->is_f16, &L.kb, err))
                return false;
            if (!upload_vec_f32(file, (p + "self_attn.v_proj.bias").c_str(), e->is_f16, &L.vb, err))
                return false;
            if (!upload_raw(file, (p + "self_attn.q_proj.weight").c_str(), &L.q, err)) return false;
            if (!upload_raw(file, (p + "self_attn.k_proj.weight").c_str(), &L.k, err)) return false;
            if (!upload_raw(file, (p + "self_attn.v_proj.weight").c_str(), &L.v, err)) return false;
            if (!upload_raw(file, (p + "self_attn.o_proj.weight").c_str(), &L.o, err)) return false;
            if (!upload_raw(file, (p + "mlp.gate_proj.weight").c_str(), &L.gate, err)) return false;
            if (!upload_raw(file, (p + "mlp.up_proj.weight").c_str(), &L.up, err)) return false;
            if (!upload_raw(file, (p + "mlp.down_proj.weight").c_str(), &L.down, err)) return false;
        }

        // 激活缓冲
        auto alloc = [&](float **p, int n) {
            ENG_CHECK(cudaMalloc(reinterpret_cast<void **>(p), static_cast<size_t>(n) * sizeof(float)));
        };
        alloc(&e->d_hidden, e->hidden);
        alloc(&e->d_normed, e->hidden);
        alloc(&e->d_q, e->q_dim);
        alloc(&e->d_k, e->kv_dim);
        alloc(&e->d_v, e->kv_dim);
        alloc(&e->d_attn, e->q_dim);
        alloc(&e->d_o, e->hidden);
        alloc(&e->d_gate, e->inter);
        alloc(&e->d_up, e->inter);
        alloc(&e->d_ffn, e->hidden);
        alloc(&e->d_logits, e->vocab);

        // KV arena（与 host KvCache 同布局，清零）
        e->layer_stride = static_cast<size_t>(e->n_kv_heads) * e->max_seq_len * e->head_dim;
        const size_t kv_floats = 2 * static_cast<size_t>(e->n_layers) * e->layer_stride;
        ENG_CHECK(cudaMalloc(reinterpret_cast<void **>(&e->d_kv), kv_floats * sizeof(float)));
        ENG_CHECK(cudaMemset(e->d_kv, 0, kv_floats * sizeof(float)));

        ENG_CHECK(cudaMalloc(reinterpret_cast<void **>(&e->d_next), sizeof(int)));

        std::fprintf(stderr,
                     "[gpu engine] 权重已常驻显存（%s），kv %.1f MB，max_seq_len=%d\n",
                     e->is_f16 ? "f16" : "f32", kv_floats * sizeof(float) / (1024.0 * 1024.0),
                     e->max_seq_len);

        *out = e;
        return true;
    }

    // ========================================================================
    // decode_step：单 stream 串起整段 forward，末尾一次 argmax D2H。
    // ========================================================================
    static int engine_step(GpuDecodeEngine *e, int token_id) {
        const int pos = e->seq_len;
        if (token_id < 0 || token_id >= e->vocab) {
            std::fprintf(stderr, "gpu engine: token_id %d 越界 [0,%d)\n", token_id, e->vocab);
            std::abort();
        }
        if (pos >= e->max_seq_len) {
            std::fprintf(stderr, "gpu engine: 位置 %d 超过 max_seq_len %d\n", pos, e->max_seq_len);
            std::abort();
        }
        cudaStream_t s = e->stream;

        // 1. 词嵌入
        gpu::embed_lookup(s, e->d_hidden, e->d_embed, token_id, e->hidden, e->is_f16);

        // 2. 逐层 transformer
        for (int i = 0; i < e->n_layers; ++i) {
            auto &L = e->layers[i];
            float *k_plane = e->d_kv + static_cast<size_t>(i) * e->layer_stride;
            float *v_plane = e->d_kv + static_cast<size_t>(e->n_layers + i) * e->layer_stride;

            gpu::rmsnorm(s, e->d_normed, e->d_hidden, L.input_ln, e->hidden, e->rms_eps);
            mv_qkv(e, L.q, L.k, L.v, e->d_normed, e->d_q, e->d_k, e->d_v, e->q_dim, e->kv_dim,
                   e->hidden);
            gpu::bias_add(s, e->d_q, L.qb, e->q_dim);
            gpu::bias_add(s, e->d_k, L.kb, e->kv_dim);
            gpu::bias_add(s, e->d_v, L.vb, e->kv_dim);
            gpu::rope(s, e->d_q, e->d_k, e->n_heads, e->n_kv_heads, e->head_dim, pos,
                      e->rope_theta);
            gpu::kv_append(s, k_plane, v_plane, e->d_k, e->d_v, pos, e->n_kv_heads,
                           e->max_seq_len, e->head_dim);
            gpu::attention_decode(s, e->d_attn, e->d_q, k_plane, v_plane, pos + 1, e->max_seq_len,
                                  e->n_heads, e->n_kv_heads, e->head_dim, e->attn_scale);
            mv(e, L.o, e->d_attn, e->d_o, e->hidden, e->q_dim);
            gpu::residual_add(s, e->d_hidden, e->d_o, e->hidden);
            gpu::rmsnorm(s, e->d_normed, e->d_hidden, L.post_ln, e->hidden, e->rms_eps);
            mv_pair(e, L.gate, L.up, e->d_normed, e->d_gate, e->d_up, e->inter, e->hidden);
            gpu::swiglu(s, e->d_gate, e->d_up, e->inter);
            mv(e, L.down, e->d_gate, e->d_ffn, e->hidden, e->inter);
            gpu::residual_add(s, e->d_hidden, e->d_ffn, e->hidden);
        }

        // 3. 最终 norm + lm_head
        gpu::rmsnorm(s, e->d_normed, e->d_hidden, e->d_final_norm, e->hidden, e->rms_eps);
        mv(e, e->d_lm_head, e->d_normed, e->d_logits, e->vocab, e->hidden);

        // 4. argmax + 唯一的 D2H
        gpu::argmax(s, e->d_logits, e->d_next, e->vocab);
        ENG_CHECK(cudaStreamSynchronize(s)); // 等整段完成，顺带捕获运行期错误
        int next = 0;
        ENG_CHECK(cudaMemcpy(&next, e->d_next, sizeof(int), cudaMemcpyDeviceToHost));

        e->seq_len += 1;
        return next;
    }

    static void engine_reset(GpuDecodeEngine *e) { e->seq_len = 0; } // 对齐 host KvCache::reset

    static void engine_destroy(GpuDecodeEngine *e) {
        if (!e) return;
        auto freep = [&](void *p) { if (p) cudaFree(p); };
        freep(e->d_embed);
        if (e->d_lm_head != e->d_embed) freep(e->d_lm_head);
        freep(e->d_final_norm);
        for (auto &L : e->layers) {
            freep(L.input_ln); freep(L.post_ln); freep(L.qb); freep(L.kb); freep(L.vb);
            freep(L.q); freep(L.k); freep(L.v); freep(L.o); freep(L.gate); freep(L.up);
            freep(L.down);
        }
        freep(e->d_hidden); freep(e->d_normed); freep(e->d_q); freep(e->d_k); freep(e->d_v);
        freep(e->d_attn); freep(e->d_o); freep(e->d_gate); freep(e->d_up); freep(e->d_ffn);
        freep(e->d_logits); freep(e->d_kv); freep(e->d_next);
        if (e->stream) cudaStreamDestroy(e->stream);
        delete e;
    }

    static const float *engine_logits(const GpuDecodeEngine *e) { return e->d_logits; }

    // 自注册进 decode-engine registry（名字 "cuda"）。仅 CUDA 构建编译到这里。
    TINYQWEN_GPU_DECODE_VARIANT("cuda", engine_create, engine_step, engine_reset, engine_destroy,
                                engine_logits);
} // namespace tinyqwen
