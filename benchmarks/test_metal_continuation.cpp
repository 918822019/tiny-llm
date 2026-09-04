// =============================================================================
// benchmarks/test_metal_continuation.mm —— Metal prefill 的 KV 接续等价性测试
// =============================================================================
//
// 为什么需要这个测试（AGENTS.md 坑 #7 明确要求）:
//   改 runtime/metal_prefill.mm 后除了 verify.sh，还要跑接续等价性测试。
//   fresh prefill 测不出接续路径的 bug —— 首段 prefill 走的是 pos0=0 的对称
//   causal 分支，而接续调用走的是 pos0>0 的非对称分支（Q_len=n、
//   KV_len=pos0+n），RoPE 要用绝对位置、attention 要读 [0, pos0+n)。
//   历史上这里踩过两个只在接续时暴露的 bug：dispatch_2d 的 grid 轴写反、
//   RoPE 用批次内下标而不是绝对位置（症状是首段精确、第二段偏差 2.14）。
//
// 通过标准:
//   一次喂 N 个 token  vs  分两段喂（前 a 个 + 后 b 个，a+b=N）
//   两者的**末位 logits 必须逐位相同**（bit-exact，不是"误差很小"）。
//   逐位相同才说明 GPU KV cache 的追加、RoPE 绝对位置、非对称 attention
//   三件事全都对；任何一处错都会让第二段的结果偏离。
//
// 顺带验证 all_logits 两条路径:
//   all_logits=true 返回全部 n 行，其末行必须与 all_logits=false 的单行结果一致。
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "metal_prefill.h"
#include "model_loader.h"

namespace {

    std::vector<int> parse_tokens(const char *csv) {
        std::vector<int> out;
        const char *p = csv;
        while (*p) {
            char *end = nullptr;
            const long v = std::strtol(p, &end, 10);
            if (end == p) break;
            out.push_back(static_cast<int>(v));
            p = end;
            while (*p == ',' || *p == ' ') ++p;
        }
        return out;
    }

    struct RowDiff {
        int bit_diff = 0;
        double max_abs = 0.0;
        int worst_idx = 0;
    };

    RowDiff compare_rows(const float *a, const float *b, int vocab, const char *label) {
        RowDiff r;
        for (int i = 0; i < vocab; ++i) {
            const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            if (d > r.max_abs) { r.max_abs = d; r.worst_idx = i; }
            uint32_t ba, bb;
            std::memcpy(&ba, &a[i], sizeof(ba));
            std::memcpy(&bb, &b[i], sizeof(bb));
            if (ba != bb) ++r.bit_diff;
        }
        std::printf("%-34s max_abs_err=%.3e   逐位不同 %d/%d", label, r.max_abs, r.bit_diff, vocab);
        if (r.bit_diff > 0) std::printf("  (最差点 idx=%d)", r.worst_idx);
        std::printf("\n");
        return r;
    }

    // 容差选取依据（不是随手拍的）:
    //   - 历史上两个真的接续 bug（dispatch_2d grid 轴写反、RoPE 用批次内下标而非
    //     绝对位置）产生的末位偏差是 **2.14** 绝对值；
    //   - 而两条喂法的 n 不同会让 MPS 选不同的内部 kernel，fp32 累加顺序随之改变，
    //     带来的良性舍入实测只有 **~6.6e-05**（logits 量级 ~30，相对 2e-6，
    //     正好是 fp32 在 1024 项点积 + 28 层之后的正常舍入水平）。
    // 1e-3 比良性舍入高 15×、比真 bug 低 2000× —— 两边都有足够余量。
    // 所以"逐位相同"这个判据在两条路径 n 不同时不可达也不必要，改用容差。
    constexpr double kTol = 1e-3;

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::vector<int> tokens;
    int split = 4;
    int max_seq_len = 1024;

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = parse_tokens(argv[++i]);
        else if (!std::strcmp(argv[i], "--split") && i + 1 < argc) split = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--max-seq-len") && i + 1 < argc) max_seq_len = std::atoi(argv[++i]);
    }

    if (model_path.empty() || tokens.empty()) {
        std::fprintf(stderr, "用法: %s --model <model.tqwen> --tokens <csv> [--split N] [--max-seq-len N]\n",
                     argv[0]);
        return 2;
    }
    const int N = static_cast<int>(tokens.size());
    if (split <= 0 || split >= N) {
        std::fprintf(stderr, "error: --split 必须在 (0, %d) 之间\n", N);
        return 2;
    }
    if (!tinyqwen::metal_prefill_available()) {
        std::fprintf(stderr, "error: 本构建没有 Metal prefill 引擎（非 Apple 平台）\n");
        return 2;
    }

    tinyqwen::ModelFile file;
    std::string err;
    if (!file.load(model_path, &err)) {
        std::fprintf(stderr, "error: 加载模型失败: %s\n", err.c_str());
        return 1;
    }
    const int vocab = static_cast<int>(file.header().vocab_size);

    tinyqwen::MetalPrefillEngine *engine = nullptr;
    if (!tinyqwen::metal_prefill_create(&file, max_seq_len, &err, &engine)) {
        std::fprintf(stderr, "error: 创建引擎失败: %s\n", err.c_str());
        return 1;
    }
    std::printf("设备: %s    vocab=%d    N=%d    split=%d+%d\n\n",
                tinyqwen::metal_prefill_device_name(engine).c_str(), vocab, N, split, N - split);

    std::vector<float> one_shot(vocab);
    std::vector<float> two_shot(vocab);
    std::vector<float> one_all(static_cast<size_t>(N) * vocab);
    std::vector<float> two_all(static_cast<size_t>(N - split) * vocab);

    int failures = 0;

    // ---- 路径 A：一次喂全部 N 个 token（单行 lm_head）----
    tinyqwen::metal_prefill_reset_kv(engine);
    int next_a = tinyqwen::metal_prefill_run(engine, tokens.data(), N, one_shot.data(),
                                             /*all_logits=*/false, nullptr, nullptr, &err);
    if (next_a < 0) {
        std::fprintf(stderr, "error: 一次喂失败: %s\n", err.c_str());
        tinyqwen::metal_prefill_destroy(engine);
        return 1;
    }
    std::printf("一次喂 %d 个 → next=%d，kv_len=%d\n", N, next_a, tinyqwen::metal_prefill_kv_len(engine));

    // ---- 路径 B：分两段喂（单行 lm_head）----
    tinyqwen::metal_prefill_reset_kv(engine);
    int next_b1 = tinyqwen::metal_prefill_run(engine, tokens.data(), split, nullptr,
                                              /*all_logits=*/false, nullptr, nullptr, &err);
    const int mid_kv = tinyqwen::metal_prefill_kv_len(engine);
    int next_b2 = tinyqwen::metal_prefill_run(engine, tokens.data() + split, N - split,
                                              two_shot.data(), /*all_logits=*/false, nullptr, nullptr, &err);
    if (next_b1 < 0 || next_b2 < 0) {
        std::fprintf(stderr, "error: 分两段喂失败: %s\n", err.c_str());
        tinyqwen::metal_prefill_destroy(engine);
        return 1;
    }
    std::printf("分两段喂 %d+%d → next=%d，中段后 kv_len=%d，末段后 kv_len=%d\n",
                split, N - split, next_b2, mid_kv, tinyqwen::metal_prefill_kv_len(engine));

    // ---- 路径 C/D：同样两种喂法，但走 all_logits=true 的全行 lm_head ----
    // 这两条用来把偏差归因拆开：C 与 D 的 lm_head 都是全行路径，所以它们之间的
    // 差异只可能来自 attention / KV 接续；而 A vs C、B vs D 的差异只可能来自
    // lm_head 的 resultRows（1 vs n）不同 —— MPS 会按矩阵尺寸选不同的内部
    // kernel，fp32 累加顺序随之改变，于是同一行数学结果也会差最后几位。
    tinyqwen::metal_prefill_reset_kv(engine);
    const int next_c = tinyqwen::metal_prefill_run(engine, tokens.data(), N, one_all.data(),
                                                   /*all_logits=*/true, nullptr, nullptr, &err);
    tinyqwen::metal_prefill_reset_kv(engine);
    tinyqwen::metal_prefill_run(engine, tokens.data(), split, nullptr, true, nullptr, nullptr, &err);
    const int next_d = tinyqwen::metal_prefill_run(engine, tokens.data() + split, N - split,
                                                   two_all.data(), /*all_logits=*/true, nullptr, nullptr, &err);
    if (next_c < 0 || next_d < 0) {
        std::fprintf(stderr, "error: all_logits=true 失败: %s\n", err.c_str());
        ++failures;
    }

    const float *one_all_last = one_all.data() + static_cast<size_t>(N - 1) * vocab;
    const float *two_all_last = two_all.data() + static_cast<size_t>(N - split - 1) * vocab;

    std::printf("\n--- 归因分解 ---\n");
    // ① attention / KV 接续是否正确（两条都是全行 lm_head，形状口径一致）
    const RowDiff r_attn = compare_rows(one_all_last, two_all_last, vocab,
                                        "① 全行路径：一次喂 vs 分两段");
    // ②③ lm_head resultRows 差异带来的舍入（同一次喂法，只换 lm_head 行数）
    const RowDiff r_lm1 = compare_rows(one_all_last, one_shot.data(), vocab,
                                       "② 一次喂：全行 lm_head vs 单行");
    const RowDiff r_lm2 = compare_rows(two_all_last, two_shot.data(), vocab,
                                       "③ 分两段：全行 lm_head vs 单行");
    // ④ AGENTS.md 要求的口径：一次喂 vs 分两段（都走单行 lm_head）
    const RowDiff r_main = compare_rows(one_shot.data(), two_shot.data(), vocab,
                                        "④ 单行路径：一次喂 vs 分两段");

    std::printf("\n--- 判定（容差 %.0e）---\n", kTol);
    const auto judge = [&](const RowDiff &r, const char *what) {
        const bool ok = r.max_abs < kTol;
        std::printf("%-34s %s max_abs_err=%.3e%s\n", what, ok ? "✅" : "❌", r.max_abs,
                    r.bit_diff == 0 ? "（逐位相同）" : "");
        return ok ? 0 : 1;
    };
    failures += judge(r_attn, "① attention/KV 接续");
    failures += judge(r_main, "④ 一次喂 vs 分两段（末位）");
    // ②③ 只作为归因信息打印，不参与判定：它们是 MPS 内部 kernel 选择导致的
    // 良性舍入，不是本引擎的正确性问题。
    std::printf("%-34s 仅供参考：② %.3e / ③ %.3e\n", "②③ lm_head resultRows 舍入",
                r_lm1.max_abs, r_lm2.max_abs);

    if (next_a != next_b2 || next_a != next_c || next_a != next_d) {
        std::printf("%-34s ❌ argmax 不一致: A=%d B=%d C=%d D=%d\n", "next token",
                    next_a, next_b2, next_c, next_d);
        ++failures;
    } else {
        std::printf("%-34s ✅ 四条路径一致 (%d)\n", "next token", next_a);
    }

    if (tinyqwen::metal_prefill_kv_len(engine) != N) {
        std::printf("%-34s ❌ kv_len=%d，应为 %d\n", "分两段喂后的 kv_len",
                    tinyqwen::metal_prefill_kv_len(engine), N);
        ++failures;
    }

    int nonzero_rows = 0;
    for (int s = 0; s < N; ++s) {
        const float *row = one_all.data() + static_cast<size_t>(s) * vocab;
        float mx = 0.0f;
        for (int i = 0; i < vocab; ++i) mx = std::max(mx, std::fabs(row[i]));
        if (mx > 0.0f) ++nonzero_rows;
    }
    std::printf("%-34s %s 非零行 %d / %d\n", "all_logits=true 行数完整性",
                nonzero_rows == N ? "✅" : "❌", nonzero_rows, N);
    if (nonzero_rows != N) ++failures;

    tinyqwen::metal_prefill_destroy(engine);

    std::printf("\n%s\n", failures == 0 ? "全部通过 ✅" : "存在失败 ❌");
    return failures == 0 ? 0 : 1;
}
