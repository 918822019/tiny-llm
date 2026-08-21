// ============================================================================
// test_gpu_ops.cpp — GPU decode engine 的算子对齐测试
// ============================================================================
// 本文件逐个验证 kernels/cuda/gpu_kernels.cu 中的 device 指针算子与对应
// *_ref 参考实现的数值一致性。是 GPU 后端的正确性门禁——先证明每个算子
// 单独算对，再谈整段 forward 的正确性。
//
// 覆盖的 GPU 算子：
//   - gpu::rmsnorm          — RMSNorm 归一化
//   - gpu::bias_add         — 偏置加法
//   - gpu::residual_add     — 残差加法
//   - gpu::rope             — 旋转位置编码
//   - gpu::swiglu           — SwiGLU 激活
//   - gpu::embed_lookup     — Embedding 查表
//   - gpu::argmax           — Argmax（logits → token ID）
//   - gpu::kv_append        — KV cache 追加
//   - gpu::attention_decode — Decode 阶段的注意力计算
//   - gpu::matvec_f32/f16   — 矩阵向量乘法
//
// 编译条件：仅在 TINYQWEN_HAS_CUDA 时编译（见 tests/CMakeLists.txt）。
// 运行条件：有 CUDA 设备才执行测试，否则 [skip]（支持无卡机器编译）。
//
// 测试模式：host 构造数据 → cudaMemcpy H2D → kernel 执行 → cudaMemcpy D2H
//          → 与 ref 逐元素比较。
// ============================================================================

#include "test_framework.h"  // 自研测试框架

#include <cmath>      // std::sqrt, std::fabs
#include <cstdint>    // uint16_t
#include <vector>     // std::vector

#include <cuda_runtime.h>  // CUDA runtime API（malloc/free/memcpy/sync）

#include "cuda/gpu_kernels.cuh"  // GPU kernel 启动器声明
#include "ref_ops.h"              // CPU 参考实现

using namespace tinyqwen;

namespace {
    // =========================================================================
    // has_device() — 检查是否有可用的 CUDA 设备
    // =========================================================================
    // 有 CUDA 构建但无 GPU 卡的机器上返回 false，测试将 skip。
    bool has_device() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) return false;  // CUDA 驱动不可用
        return n > 0;  // 至少有一个设备
    }

    // =========================================================================
    // rnd() — 确定性伪随机数生成
    // =========================================================================
    // 基于简单哈希：(i * 37 % mod) - mod/2，再乘以 scale。
    // 不依赖 <random>，保证跨平台可复现。
    float rnd(int i, int mod, float scale) {
        return static_cast<float>((i * 37 % mod) - mod / 2) * scale;
    }

    // =========================================================================
    // struct DBuf — device 缓冲 RAII 包装
    // =========================================================================
    // 构造时 cudaMalloc，析构时 cudaFree，防止内存泄漏。
    // as<T>() 提供类型安全的指针转换。
    struct DBuf {
        void *p = nullptr;                      // device 端原始指针
        explicit DBuf(size_t bytes) { cudaMalloc(&p, bytes); }  // 分配 device 内存
        ~DBuf() { if (p) cudaFree(p); }         // 释放 device 内存
        template <typename T> T *as() { return static_cast<T *>(p); }  // 类型转换
    };
} // namespace

// ---------------------------------------------------------------------------
// 测试：gpu::rmsnorm vs rmsnorm_ref
//
// n=896（真实 hidden_dim），构造随机 x 和 w。
// host 跑 ref，device 跑 gpu kernel，逐元素对比。
// 容差 1e-4：GPU float32 累加 vs CPU double 累加的精度差异。
// ---------------------------------------------------------------------------
TEST (gpu_rmsnorm_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 896;  // 真实模型 hidden dimension
    std::vector<float> x(n), w(n), y_ref(n), y_gpu(n);
    // 填充确定性随机数据
    for (int i = 0; i < n; ++i) { x[i] = rnd(i, 29, 0.25f); w[i] = rnd(i + 7, 23, 0.125f) + 1.0f; }
    // CPU 参考结果
    rmsnorm_ref(x.data(), w.data(), y_ref.data(), n, 1e-6f);
    // 分配 device 缓冲并上传数据
    DBuf dx(n * sizeof(float)), dw(n * sizeof(float)), dy(n * sizeof(float));
    cudaMemcpy(dx.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    // 执行 GPU kernel（stream=nullptr 表示默认流）
    gpu::rmsnorm(nullptr, dy.as<float>(), dx.as<float>(), dw.as<float>(), n, 1e-6f);
    cudaDeviceSynchronize();  // 等待 GPU 完成
    // 下载结果并逐元素对比
    cudaMemcpy(y_gpu.data(), dy.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(y_gpu[i], y_ref[i], 1e-4);
}

// ---------------------------------------------------------------------------
// 测试：gpu::bias_add 和 gpu::residual_add
//
// 两者语义相同：out[i] = a[i] + bias[i]。
// n=896，验证两种加法 kernel 都与期望值一致。
// ---------------------------------------------------------------------------
TEST (gpu_bias_residual_match) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 896;
    std::vector<float> a(n), bias(n), got(n);
    for (int i = 0; i < n; ++i) { a[i] = rnd(i, 31, 0.5f); bias[i] = rnd(i + 3, 17, 0.25f); }
    DBuf da(n * sizeof(float)), db(n * sizeof(float));

    // ---- bias_add 测试 ----
    cudaMemcpy(da.p, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db.p, bias.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::bias_add(nullptr, da.as<float>(), db.as<float>(), n);  // da += db
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), da.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(got[i], a[i] + bias[i], 1e-5);

    // ---- residual_add 测试 ----
    cudaMemcpy(da.p, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::residual_add(nullptr, da.as<float>(), db.as<float>(), n);  // da += db
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), da.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(got[i], a[i] + bias[i], 1e-5);
}

// ---------------------------------------------------------------------------
// 测试：gpu::rope vs rope_ref
//
// GQA 配置：14 q heads / 2 kv heads，head_dim=64。
// 在多个 pos（0, 1, 7, 100）下验证旋转位置编码的正确性。
// pos=0 时应为恒等；pos>0 时 Q/K 各维度按频率旋转。
// ---------------------------------------------------------------------------
TEST (gpu_rope_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_heads = 14, n_kv_heads = 2, head_dim = 64;
    const int qn = n_heads * head_dim, kn = n_kv_heads * head_dim;  // Q/K 总维度
    std::vector<float> q(qn), k(kn);
    for (int i = 0; i < qn; ++i) q[i] = rnd(i, 29, 0.5f);
    for (int i = 0; i < kn; ++i) k[i] = rnd(i + 11, 23, 0.5f);

    // 在多个位置验证
    for (int pos : {0, 1, 7, 100}) {
        std::vector<float> qr = q, kr = k, qg = q, kg = k;  // 复制输入
        // CPU 参考
        rope_ref(qr.data(), kr.data(), n_heads, n_kv_heads, head_dim, pos, 10000.0f);
        // GPU kernel
        DBuf dq(qn * sizeof(float)), dk(kn * sizeof(float)), dpos(sizeof(int));
        cudaMemcpy(dq.p, qg.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dk.p, kg.data(), kn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
        gpu::rope(nullptr, dq.as<float>(), dk.as<float>(), n_heads, n_kv_heads, head_dim,
                  dpos.as<int>(), 10000.0f);
        cudaDeviceSynchronize();
        cudaMemcpy(qg.data(), dq.p, qn * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(kg.data(), dk.p, kn * sizeof(float), cudaMemcpyDeviceToHost);
        // 逐元素对比 Q 和 K
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(qg[i], qr[i], 1e-4);
        for (int i = 0; i < kn; ++i) EXPECT_NEAR(kg[i], kr[i], 1e-4);
    }
}

// ---------------------------------------------------------------------------
// 测试：gpu::swiglu vs swiglu_ref
//
// n=4864（真实 FFN intermediate dim），验证 SwiGLU 融合算子。
// SwiGLU(gate, up) = silu(gate) * up，就地修改 gate。
// ---------------------------------------------------------------------------
TEST (gpu_swiglu_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 4864;  // FFN intermediate dimension
    std::vector<float> gate(n), up(n), g_ref(n), g_gpu(n);
    for (int i = 0; i < n; ++i) { gate[i] = rnd(i, 29, 0.5f); up[i] = rnd(i + 5, 23, 0.5f); }
    g_ref = gate;  // 复制一份给 ref
    swiglu_ref(g_ref.data(), up.data(), n);  // CPU 参考
    // GPU kernel
    DBuf dg(n * sizeof(float)), du(n * sizeof(float));
    cudaMemcpy(dg.p, gate.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du.p, up.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::swiglu(nullptr, dg.as<float>(), du.as<float>(), n);
    cudaDeviceSynchronize();
    cudaMemcpy(g_gpu.data(), dg.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(g_gpu[i], g_ref[i], 1e-4);
}

// ---------------------------------------------------------------------------
// 测试：gpu::embed_lookup vs host 手动查表
//
// vocab=1000, hidden=896, token=377。
// 验证 embedding 查表 kernel 能正确从大表中取出指定行的向量。
// ---------------------------------------------------------------------------
TEST (gpu_embed_matches_host) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int vocab = 1000, hidden = 896, token = 377;
    std::vector<float> embed((size_t)vocab * hidden), h_ref(hidden), h_gpu(hidden);
    for (size_t i = 0; i < embed.size(); ++i) embed[i] = rnd((int)i, 29, 0.125f);
    // host 参照：直接从 embedding 表中拷贝第 token 行
    for (int j = 0; j < hidden; ++j) h_ref[j] = embed[(size_t)token * hidden + j];
    // GPU kernel
    DBuf de(embed.size() * sizeof(float)), dh(hidden * sizeof(float)), dtok(sizeof(int));
    cudaMemcpy(de.p, embed.data(), embed.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dtok.p, &token, sizeof(int), cudaMemcpyHostToDevice);
    gpu::embed_lookup(nullptr, dh.as<float>(), de.p, dtok.as<int>(), hidden, false);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu.data(), dh.p, hidden * sizeof(float), cudaMemcpyDeviceToHost);
    for (int j = 0; j < hidden; ++j) EXPECT_NEAR(h_gpu[j], h_ref[j], 1e-5);
}

// ---------------------------------------------------------------------------
// 测试：gpu::argmax vs argmax_ref
//
// 覆盖多种场景：常规随机 logits、唯一最大值、平局取首个、全相等。
// vocab_size=151936（Qwen2.5 真实词表大小）。
// ---------------------------------------------------------------------------
TEST (gpu_argmax_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    // lambda：封装一次 argmax GPU 调用
    auto run = [](const std::vector<float> &logits) {
        std::vector<int> got(1, -1);
        DBuf dl(logits.size() * sizeof(float)), di(sizeof(int));
        cudaMemcpy(dl.p, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
        gpu::argmax(nullptr, dl.as<float>(), di.as<int>(), (int)logits.size());
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), di.p, sizeof(int), cudaMemcpyDeviceToHost);
        return got[0];
    };
    // 场景 1：常规随机 logits
    std::vector<float> a(151936);
    for (size_t i = 0; i < a.size(); ++i) a[i] = rnd((int)i, 41, 0.1f);
    EXPECT_EQ(run(a), argmax_ref(a.data(), (int)a.size()));
    // 场景 2：唯一最大值在已知位置
    a[12345] = 99.0f;
    EXPECT_EQ(run(a), 12345);
    // 场景 3：平局取首个（两个 3.0，应返回索引 1）
    std::vector<float> tie = {0.1f, 3.0f, -2.0f, 3.0f, 1.5f};
    EXPECT_EQ(run(tie), argmax_ref(tie.data(), 5));
    // 场景 4：全相等 → 返回 0
    std::vector<float> first = {5.0f, 5.0f, 5.0f};
    EXPECT_EQ(run(first), 0);
}

// ---------------------------------------------------------------------------
// 测试：gpu::kv_append vs host 手动索引写入
//
// 验证 KV cache 追加 kernel 将新 token 的 K/V 写入正确的槽位。
// n_kv_heads=2, max_seq_len=8, head_dim=64, pos=5。
// host 参照：按 qwen_model 的平面布局公式计算目标地址。
// ---------------------------------------------------------------------------
TEST (gpu_kv_append_matches_host) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_kv_heads = 2, max_seq_len = 8, head_dim = 64;
    const int per = n_kv_heads * head_dim;                          // 每个 token 的 K/V 元素数
    const size_t plane = (size_t)n_kv_heads * max_seq_len * head_dim;  // 一个平面的大小
    std::vector<float> k_in(per), v_in(per);
    for (int i = 0; i < per; ++i) { k_in[i] = rnd(i, 29, 0.5f); v_in[i] = rnd(i + 9, 23, 0.5f); }
    const int pos = 5;  // 写入位置

    // host 参照：按平面布局公式计算目标槽位
    std::vector<float> kh(plane, 0.0f), vh(plane, 0.0f);
    const size_t head_plane = (size_t)max_seq_len * head_dim;  // 每个头的平面大小
    for (int h = 0; h < n_kv_heads; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const size_t dst = (size_t)h * head_plane + (size_t)pos * head_dim + d;
            kh[dst] = k_in[h * head_dim + d];
            vh[dst] = v_in[h * head_dim + d];
        }

    // GPU kernel
    DBuf dkp(plane * sizeof(float)), dvp(plane * sizeof(float)), dki(per * sizeof(float)),
            dvi(per * sizeof(float)), dpos(sizeof(int));
    cudaMemset(dkp.p, 0, plane * sizeof(float));  // device plane 清零
    cudaMemset(dvp.p, 0, plane * sizeof(float));
    cudaMemcpy(dki.p, k_in.data(), per * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvi.p, v_in.data(), per * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
    gpu::kv_append(nullptr, dkp.as<float>(), dvp.as<float>(), dki.as<float>(), dvi.as<float>(),
                   dpos.as<int>(), n_kv_heads, max_seq_len, head_dim);
    cudaDeviceSynchronize();

    // 下载结果并验证：目标槽位一致，其余位置仍为 0
    std::vector<float> kg(plane), vg(plane);
    cudaMemcpy(kg.data(), dkp.p, plane * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vg.data(), dvp.p, plane * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < plane; ++i) {
        EXPECT_NEAR(kg[i], kh[i], 1e-6);
        EXPECT_NEAR(vg[i], vh[i], 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 测试：gpu::attention_decode vs attention_decode_ref
//
// GQA 配置：14 q heads / 2 kv heads, head_dim=64, max_seq_len=16。
// 在多个 seq_len（1, 3, 16）下验证 decode 阶段注意力的正确性。
// 容差 5e-3：attention 涉及 softmax + 加权求和，累积误差较大。
// ---------------------------------------------------------------------------
TEST (gpu_attention_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_heads = 14, n_kv_heads = 2, head_dim = 64, max_seq_len = 16;
    const int qn = n_heads * head_dim;
    const size_t plane = (size_t)n_kv_heads * max_seq_len * head_dim;
    std::vector<float> q(qn), kc(plane), vc(plane);
    for (int i = 0; i < qn; ++i) q[i] = rnd(i, 29, 0.5f);
    for (size_t i = 0; i < plane; ++i) { kc[i] = rnd((int)i, 31, 0.3f); vc[i] = rnd((int)i + 7, 23, 0.3f); }
    const float scale = 1.0f / std::sqrt((float)head_dim);  // attention score 缩放因子

    for (int seq_len : {1, 3, 16}) {
        std::vector<float> o_ref(qn), o_gpu(qn);
        // CPU 参考
        attention_decode_ref(q.data(), kc.data(), vc.data(), seq_len, max_seq_len, n_heads,
                             n_kv_heads, head_dim, scale, o_ref.data());
        // GPU kernel
        DBuf dq(qn * sizeof(float)), dkc(plane * sizeof(float)), dvc(plane * sizeof(float)),
                dout(qn * sizeof(float)), dpos(sizeof(int));
        cudaMemcpy(dq.p, q.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dkc.p, kc.data(), plane * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dvc.p, vc.data(), plane * sizeof(float), cudaMemcpyHostToDevice);
        const int pos = seq_len - 1;  // attention_decode 现在吃 pos，内部算 seq_len=pos+1
        cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
        gpu::attention_decode(nullptr, dout.as<float>(), dq.as<float>(), dkc.as<float>(),
                              dvc.as<float>(), dpos.as<int>(), max_seq_len, n_heads, n_kv_heads,
                              head_dim, scale);
        cudaDeviceSynchronize();
        cudaMemcpy(o_gpu.data(), dout.p, qn * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(o_gpu[i], o_ref[i], 5e-3);
    }
}

// ---------------------------------------------------------------------------
// 测试：gpu::matvec_f32 和 gpu::matvec_f16 vs ref
//
// out_dim=128, in_dim=896。分别验证 f32 和 f16 权重的 matvec kernel。
// f16 测试：先将 f32 权重转成 half，再用 matvec_f16_ref 作参照。
// ---------------------------------------------------------------------------
TEST (gpu_matvec_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int out_dim = 128, in_dim = 896;
    std::vector<float> w((size_t)out_dim * in_dim), x(in_dim), y_ref(out_dim), y_gpu(out_dim);
    for (size_t i = 0; i < w.size(); ++i) w[i] = rnd((int)i, 29, 0.125f);
    for (int i = 0; i < in_dim; ++i) x[i] = rnd(i, 17, 0.25f);

    // ---- f32 single matvec ----
    matvec_f32_ref(w.data(), x.data(), y_ref.data(), out_dim, in_dim);
    {
        DBuf dw(w.size() * sizeof(float)), dx(in_dim * sizeof(float)), dy(out_dim * sizeof(float));
        cudaMemcpy(dw.p, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dx.p, x.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice);
        gpu::matvec_f32(nullptr, dw.as<float>(), dx.as<float>(), dy.as<float>(), out_dim, in_dim);
        cudaDeviceSynchronize();
        cudaMemcpy(y_gpu.data(), dy.p, out_dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_gpu[o], y_ref[o], 5e-3);
    }

    // ---- f16 single matvec ----
    // 权重先量化成 half，对齐 matvec_f16_ref 的输入格式
    std::vector<uint16_t> wh(w.size());
    for (size_t i = 0; i < w.size(); ++i) wh[i] = float_to_half(w[i]);
    matvec_f16_ref(wh.data(), x.data(), y_ref.data(), out_dim, in_dim);
    {
        DBuf dw(wh.size() * sizeof(uint16_t)), dx(in_dim * sizeof(float)),
                dy(out_dim * sizeof(float));
        cudaMemcpy(dw.p, wh.data(), wh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dx.p, x.data(), in_dim * sizeof(float), cudaMemcpyHostToDevice);
        gpu::matvec_f16(nullptr, dw.as<uint16_t>(), dx.as<float>(), dy.as<float>(), out_dim,
                        in_dim);
        cudaDeviceSynchronize();
        cudaMemcpy(y_gpu.data(), dy.p, out_dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int o = 0; o < out_dim; ++o) EXPECT_NEAR(y_gpu[o], y_ref[o], 5e-3);
    }
}
