// GPU decode engine 的算子对齐测试：逐个把 kernels/cuda/gpu_kernels.cu 的
// device 指针算子与对应 *_ref 对齐（正确性门禁，先证明算对再谈整段 forward）。
// 本文件仅在 TINYQWEN_HAS_CUDA 时编译（见 tests/CMakeLists.txt）。

#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda/gpu_kernels.cuh"
#include "ref_ops.h"

using namespace tinyqwen;

namespace {
    // 运行时有可用的 CUDA 设备才测（有 CUDA 构建但无卡的机器上 skip）。
    bool has_device() {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess) return false;
        return n > 0;
    }
    // 确定性伪随机（与现有 matvec 测试同款思路，可复现）。
    float rnd(int i, int mod, float scale) {
        return static_cast<float>((i * 37 % mod) - mod / 2) * scale;
    }
    // device 缓冲包装：RAII，cudaMalloc/cudaFree。
    struct DBuf {
        void *p = nullptr;
        explicit DBuf(size_t bytes) { cudaMalloc(&p, bytes); }
        ~DBuf() { if (p) cudaFree(p); }
        template <typename T> T *as() { return static_cast<T *>(p); }
    };
} // namespace

TEST (gpu_rmsnorm_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 896;
    std::vector<float> x(n), w(n), y_ref(n), y_gpu(n);
    for (int i = 0; i < n; ++i) { x[i] = rnd(i, 29, 0.25f); w[i] = rnd(i + 7, 23, 0.125f) + 1.0f; }
    rmsnorm_ref(x.data(), w.data(), y_ref.data(), n, 1e-6f);
    DBuf dx(n * sizeof(float)), dw(n * sizeof(float)), dy(n * sizeof(float));
    cudaMemcpy(dx.p, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dw.p, w.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::rmsnorm(nullptr, dy.as<float>(), dx.as<float>(), dw.as<float>(), n, 1e-6f);
    cudaDeviceSynchronize();
    cudaMemcpy(y_gpu.data(), dy.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(y_gpu[i], y_ref[i], 1e-4);
}

TEST (gpu_bias_residual_match) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 896;
    std::vector<float> a(n), bias(n), got(n);
    for (int i = 0; i < n; ++i) { a[i] = rnd(i, 31, 0.5f); bias[i] = rnd(i + 3, 17, 0.25f); }
    DBuf da(n * sizeof(float)), db(n * sizeof(float));
    // bias_add
    cudaMemcpy(da.p, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db.p, bias.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::bias_add(nullptr, da.as<float>(), db.as<float>(), n);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), da.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(got[i], a[i] + bias[i], 1e-5);
    // residual_add
    cudaMemcpy(da.p, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::residual_add(nullptr, da.as<float>(), db.as<float>(), n);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), da.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(got[i], a[i] + bias[i], 1e-5);
}

TEST (gpu_rope_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_heads = 14, n_kv_heads = 2, head_dim = 64;
    const int qn = n_heads * head_dim, kn = n_kv_heads * head_dim;
    std::vector<float> q(qn), k(kn);
    for (int i = 0; i < qn; ++i) q[i] = rnd(i, 29, 0.5f);
    for (int i = 0; i < kn; ++i) k[i] = rnd(i + 11, 23, 0.5f);
    for (int pos : {0, 1, 7, 100}) {
        std::vector<float> qr = q, kr = k, qg = q, kg = k;
        rope_ref(qr.data(), kr.data(), n_heads, n_kv_heads, head_dim, pos, 10000.0f);
        DBuf dq(qn * sizeof(float)), dk(kn * sizeof(float)), dpos(sizeof(int));
        cudaMemcpy(dq.p, qg.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dk.p, kg.data(), kn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
        gpu::rope(nullptr, dq.as<float>(), dk.as<float>(), n_heads, n_kv_heads, head_dim,
                  dpos.as<int>(), 10000.0f);
        cudaDeviceSynchronize();
        cudaMemcpy(qg.data(), dq.p, qn * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(kg.data(), dk.p, kn * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(qg[i], qr[i], 1e-4);
        for (int i = 0; i < kn; ++i) EXPECT_NEAR(kg[i], kr[i], 1e-4);
    }
}

TEST (gpu_swiglu_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n = 4864;
    std::vector<float> gate(n), up(n), g_ref(n), g_gpu(n);
    for (int i = 0; i < n; ++i) { gate[i] = rnd(i, 29, 0.5f); up[i] = rnd(i + 5, 23, 0.5f); }
    g_ref = gate;
    swiglu_ref(g_ref.data(), up.data(), n);
    DBuf dg(n * sizeof(float)), du(n * sizeof(float));
    cudaMemcpy(dg.p, gate.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du.p, up.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    gpu::swiglu(nullptr, dg.as<float>(), du.as<float>(), n);
    cudaDeviceSynchronize();
    cudaMemcpy(g_gpu.data(), dg.p, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) EXPECT_NEAR(g_gpu[i], g_ref[i], 1e-4);
}

TEST (gpu_embed_matches_host) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int vocab = 1000, hidden = 896, token = 377;
    std::vector<float> embed((size_t)vocab * hidden), h_ref(hidden), h_gpu(hidden);
    for (size_t i = 0; i < embed.size(); ++i) embed[i] = rnd((int)i, 29, 0.125f);
    for (int j = 0; j < hidden; ++j) h_ref[j] = embed[(size_t)token * hidden + j];
    DBuf de(embed.size() * sizeof(float)), dh(hidden * sizeof(float)), dtok(sizeof(int));
    cudaMemcpy(de.p, embed.data(), embed.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dtok.p, &token, sizeof(int), cudaMemcpyHostToDevice);
    gpu::embed_lookup(nullptr, dh.as<float>(), de.p, dtok.as<int>(), hidden, false);
    cudaDeviceSynchronize();
    cudaMemcpy(h_gpu.data(), dh.p, hidden * sizeof(float), cudaMemcpyDeviceToHost);
    for (int j = 0; j < hidden; ++j) EXPECT_NEAR(h_gpu[j], h_ref[j], 1e-5);
}

TEST (gpu_argmax_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    // 常规 + 平局（取首个）+ 最大值在首/尾。
    auto run = [](const std::vector<float> &logits) {
        std::vector<int> got(1, -1);
        DBuf dl(logits.size() * sizeof(float)), di(sizeof(int));
        cudaMemcpy(dl.p, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
        gpu::argmax(nullptr, dl.as<float>(), di.as<int>(), (int)logits.size());
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), di.p, sizeof(int), cudaMemcpyDeviceToHost);
        return got[0];
    };
    std::vector<float> a(151936);
    for (size_t i = 0; i < a.size(); ++i) a[i] = rnd((int)i, 41, 0.1f);
    EXPECT_EQ(run(a), argmax_ref(a.data(), (int)a.size()));
    a[12345] = 99.0f; // 唯一最大
    EXPECT_EQ(run(a), 12345);
    std::vector<float> tie = {0.1f, 3.0f, -2.0f, 3.0f, 1.5f}; // 平局取首个
    EXPECT_EQ(run(tie), argmax_ref(tie.data(), 5));
    std::vector<float> first = {5.0f, 5.0f, 5.0f}; // 全相等 → 0
    EXPECT_EQ(run(first), 0);
}

TEST (gpu_kv_append_matches_host) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_kv_heads = 2, max_seq_len = 8, head_dim = 64;
    const int per = n_kv_heads * head_dim;
    const size_t plane = (size_t)n_kv_heads * max_seq_len * head_dim;
    std::vector<float> k_in(per), v_in(per);
    for (int i = 0; i < per; ++i) { k_in[i] = rnd(i, 29, 0.5f); v_in[i] = rnd(i + 9, 23, 0.5f); }
    const int pos = 5;
    // host 参照：按 qwen_model 的索引算出目标槽位。
    std::vector<float> kh(plane, 0.0f), vh(plane, 0.0f);
    const size_t head_plane = (size_t)max_seq_len * head_dim;
    for (int h = 0; h < n_kv_heads; ++h)
        for (int d = 0; d < head_dim; ++d) {
            const size_t dst = (size_t)h * head_plane + (size_t)pos * head_dim + d;
            kh[dst] = k_in[h * head_dim + d];
            vh[dst] = v_in[h * head_dim + d];
        }
    DBuf dkp(plane * sizeof(float)), dvp(plane * sizeof(float)), dki(per * sizeof(float)),
            dvi(per * sizeof(float)), dpos(sizeof(int));
    // device plane 清零，k_in/v_in 上传，kernel 写入 pos 槽位。
    cudaMemset(dkp.p, 0, plane * sizeof(float));
    cudaMemset(dvp.p, 0, plane * sizeof(float));
    cudaMemcpy(dki.p, k_in.data(), per * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dvi.p, v_in.data(), per * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
    gpu::kv_append(nullptr, dkp.as<float>(), dvp.as<float>(), dki.as<float>(), dvi.as<float>(),
                   dpos.as<int>(), n_kv_heads, max_seq_len, head_dim);
    cudaDeviceSynchronize();
    std::vector<float> kg(plane), vg(plane);
    cudaMemcpy(kg.data(), dkp.p, plane * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(vg.data(), dvp.p, plane * sizeof(float), cudaMemcpyDeviceToHost);
    // 目标槽位一致，其余位置仍为 0。
    for (size_t i = 0; i < plane; ++i) {
        EXPECT_NEAR(kg[i], kh[i], 1e-6);
        EXPECT_NEAR(vg[i], vh[i], 1e-6);
    }
}

TEST (gpu_attention_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int n_heads = 14, n_kv_heads = 2, head_dim = 64, max_seq_len = 16;
    const int qn = n_heads * head_dim;
    const size_t plane = (size_t)n_kv_heads * max_seq_len * head_dim;
    std::vector<float> q(qn), kc(plane), vc(plane);
    for (int i = 0; i < qn; ++i) q[i] = rnd(i, 29, 0.5f);
    for (size_t i = 0; i < plane; ++i) { kc[i] = rnd((int)i, 31, 0.3f); vc[i] = rnd((int)i + 7, 23, 0.3f); }
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for (int seq_len : {1, 3, 16}) {
        std::vector<float> o_ref(qn), o_gpu(qn);
        attention_decode_ref(q.data(), kc.data(), vc.data(), seq_len, max_seq_len, n_heads,
                             n_kv_heads, head_dim, scale, o_ref.data());
        DBuf dq(qn * sizeof(float)), dkc(plane * sizeof(float)), dvc(plane * sizeof(float)),
                dout(qn * sizeof(float)), dpos(sizeof(int));
        cudaMemcpy(dq.p, q.data(), qn * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dkc.p, kc.data(), plane * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dvc.p, vc.data(), plane * sizeof(float), cudaMemcpyHostToDevice);
        const int pos = seq_len - 1; // attention_decode 现在吃 pos，内部算 seq_len=pos+1
        cudaMemcpy(dpos.p, &pos, sizeof(int), cudaMemcpyHostToDevice);
        gpu::attention_decode(nullptr, dout.as<float>(), dq.as<float>(), dkc.as<float>(),
                              dvc.as<float>(), dpos.as<int>(), max_seq_len, n_heads, n_kv_heads,
                              head_dim, scale);
        cudaDeviceSynchronize();
        cudaMemcpy(o_gpu.data(), dout.p, qn * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < qn; ++i) EXPECT_NEAR(o_gpu[i], o_ref[i], 5e-3);
    }
}

TEST (gpu_matvec_matches_ref) {
    if (!has_device()) { std::printf("[skip] no CUDA device\n"); return; }
    const int out_dim = 128, in_dim = 896;
    std::vector<float> w((size_t)out_dim * in_dim), x(in_dim), y_ref(out_dim), y_gpu(out_dim);
    for (size_t i = 0; i < w.size(); ++i) w[i] = rnd((int)i, 29, 0.125f);
    for (int i = 0; i < in_dim; ++i) x[i] = rnd(i, 17, 0.25f);
    // f32 single
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
    // f16 single（权重先量化成 half，对齐 matvec_f16_ref）
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
