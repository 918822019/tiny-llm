// tinyqwen CLI：命令行入口，把"加载模型 -> 前向 -> 输出"串起来。
//
//   tinyqwen --model model.tqwen \
//            --tokens-json prompt_tokens.json \
//            --max-new-tokens 16 \
//            --profile-out profile.json
//
// 推理分两个阶段（概念见 docs/infra_primer.md 第 2 节）：
//   - prefill：把输入的每个 prompt token 依次喂进模型，填满 KV cache，
//     最后一步产出第一个生成 token；
//   - decode：之后每步把上一步生成的 token 再喂回去，生成下一个，循环。
//
// token ids 来自 Python（tools/tokenize_prompt.py）：v1 的 C++ 侧刻意不内置
// tokenizer（分词与研究主线无关，复用现成工具即可）。

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "dispatch.h"
#include "model_loader.h"
#include "profiler.h"
#include "qwen_model.h"

namespace {
    struct Args {
        std::string model;
        std::string tokens_csv;
        std::string tokens_json;
        // 批量模式（数据集测试）：JSONL 每行一个 {"tokens":[...]}，一次进程
        // 逐条独立 prefill+decode；--batch-out 写每条的 TTFT/decode 时序 JSON。
        std::string batch_tokens_jsonl;
        std::string batch_out;
        std::string profile_out;
        std::string dump_logits;
        int max_new_tokens = 16;
        int max_seq_len = 1024;
        int topk = 0;
        // 停止符：-2 = 按模型自动（v2 文件头带 eos_token_id 就用它，否则
        // Qwen2.5 的 im_end 151645）；-1 = 禁用；其余值 = 显式指定。
        int eos = -2;
        bool verbose = false;
        bool no_fuse_gate_up = false;
        bool no_fuse_qkv = false;
        std::string config; // 配置文件路径（可选）
        std::string matvec_impl; // matvec 实现；空 = 未指定，交给配置/默认值
        std::string ops_impl; // 非 matvec 算子实现；空 = 未指定，交给配置/默认值
        std::string engine; // decode engine；空 = CPU forward（默认），"cuda" = GPU 常驻
    };

    void usage(const char *prog) {
        std::fprintf(stderr,
                     "usage: %s --model <model.tqwen> [options]\n"
                     "  --tokens CSV            comma separated token ids\n"
                     "  --tokens-json PATH      JSON with a \"tokens\" array (tokenize_prompt.py)\n"
                     "  --batch-tokens-jsonl PATH\n"
                     "                          batch mode (dataset testing): JSONL, each line a\n"
                     "                          {\"tokens\": [...]} prompt; run sequentially with\n"
                     "                          per-prompt reset; exclusive with the two above\n"
                     "  --batch-out PATH        write per-prompt timing JSON (required w/ batch)\n"
                     "  --max-new-tokens N      default 16\n"
                     "  --max-seq-len N         KV capacity, default 1024\n"
                     "  --topk K                emit top-k logits lines (each describes the next\n"
                     "                          gen token; 0 = off)\n"
                     "  --dump-logits PATH      dump full logits (fp32 binary) after every forward\n"
                     "  --profile-out PATH      write profiler JSON\n"
                     "  --eos ID                stop token, -1 disables; default = model's own\n"
                     "                          eos (v2 header) or 151645 for v1 files\n"
                     "  --config PATH           key=value config file (CLI flags override it)\n"
                     "  --matvec-impl NAME      matvec kernel: ref (default; neon later)\n"
                     "  --ops-impl NAME         non-matvec ops (rmsnorm/rope/attention/swiglu/\n"
                     "                          argmax): ref (default) / neon\n"
                     "  --engine NAME           decode engine: '' = CPU forward (default) / cuda\n"
                     "                          (GPU-resident whole-forward; requires CUDA build)\n"
                     "  --verbose               model summary + per-token details\n",
                     prog);
    }

    bool parse_args(int argc, char **argv, Args *out) {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            // 取 flag 的值；缺值直接报错退出。
            const auto value = [&](const char *flag) -> std::string {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "error: %s needs a value\n", flag);
                    std::exit(2);
                }
                return argv[++i];
            };
            if (a == "--model") out->model = value("--model");
            else if (a == "--tokens") out->tokens_csv = value("--tokens");
            else if (a == "--tokens-json") out->tokens_json = value("--tokens-json");
            else if (a == "--batch-tokens-jsonl") out->batch_tokens_jsonl = value("--batch-tokens-jsonl");
            else if (a == "--batch-out") out->batch_out = value("--batch-out");
            else if (a == "--max-new-tokens") out->max_new_tokens = std::atoi(value("--max-new-tokens").c_str());
            else if (a == "--max-seq-len") out->max_seq_len = std::atoi(value("--max-seq-len").c_str());
            else if (a == "--topk") out->topk = std::atoi(value("--topk").c_str());
            else if (a == "--profile-out") out->profile_out = value("--profile-out");
            else if (a == "--dump-logits") out->dump_logits = value("--dump-logits");
            else if (a == "--eos") out->eos = std::atoi(value("--eos").c_str());
            else if (a == "--config") out->config = value("--config");
            else if (a == "--matvec-impl") out->matvec_impl = value("--matvec-impl");
            else if (a == "--ops-impl") out->ops_impl = value("--ops-impl");
            else if (a == "--engine") out->engine = value("--engine");
            else if (a == "--no-fuse-gate-up") out->no_fuse_gate_up = true;
            else if (a == "--no-fuse-qkv") out->no_fuse_qkv = true;
            else if (a == "--verbose") out->verbose = true;
            else if (a == "--help" || a == "-h") {
                usage(argv[0]);
                std::exit(0);
            } else {
                std::fprintf(stderr, "error: unknown flag %s\n", a.c_str());
                usage(argv[0]);
                return false;
            }
        }
        if (out->model.empty()) {
            std::fprintf(stderr, "error: --model is required\n");
            return false;
        }
        // 三种输入模式必须且只能提供一个。
        const bool has_batch = !out->batch_tokens_jsonl.empty();
        const int modes = (!out->tokens_csv.empty()) + (!out->tokens_json.empty()) + has_batch;
        if (modes != 1) {
            std::fprintf(stderr, "error: provide exactly one of "
                                 "--tokens / --tokens-json / --batch-tokens-jsonl\n");
            return false;
        }
        if (has_batch) {
            // batch 的输出契约是纯时序测量：混入 topk/logits 行会让 stdout 不
            // 可解析；engine 没有批量 prefill 入口；op 级 profiler 跨 prompt
            // 累积会爆内存（TTFT/decode 在 batch 循环里用 steady_clock 手测）。
            if (out->topk > 0 || !out->dump_logits.empty() || out->verbose ||
                !out->engine.empty() || !out->profile_out.empty()) {
                std::fprintf(stderr, "error: --batch-tokens-jsonl cannot be combined with "
                                     "--topk / --dump-logits / --verbose / --engine / --profile-out\n");
                return false;
            }
            if (out->batch_out.empty()) {
                std::fprintf(stderr, "error: --batch-tokens-jsonl requires --batch-out\n");
                return false;
            }
        }
        return true;
    }

    // "--tokens 1,2,3" -> ids。容忍空格；遇到非法字符直接报错。
    std::vector<int> parse_csv(const std::string &s) {
        std::vector<int> ids;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && (s[i] == ',' || s[i] == ' ')) ++i;
            if (i >= s.size()) break;
            size_t j = i;
            while (j < s.size() && (isdigit(static_cast<unsigned char>(s[j])) || s[j] == '-')) ++j;
            if (j == i) {
                std::fprintf(stderr, "error: bad token list near offset %zu\n", i);
                std::exit(2);
            }
            ids.push_back(std::atoi(s.substr(i, j - i).c_str()));
            i = j;
        }
        return ids;
    }

    // 从 JSON 文本中做最小化的 "tokens" 整数数组提取；不引 JSON 依赖。
    // 约定见 tools/tokenize_prompt.py 的输出格式。what 用于报错定位
    // （文件路径或 "<path> line N"）。
    std::vector<int> parse_tokens_from_string(const std::string &text, const char *what) {
        size_t key = text.find("\"tokens\"");
        if (key == std::string::npos) {
            std::fprintf(stderr, "error: no \"tokens\" field in %s\n", what);
            std::exit(2);
        }
        size_t open = text.find('[', key);
        size_t close = text.find(']', open);
        if (open == std::string::npos || close == std::string::npos) {
            std::fprintf(stderr, "error: malformed tokens array in %s\n", what);
            std::exit(2);
        }
        std::vector<int> ids;
        size_t i = open + 1;
        while (i < close) {
            while (i < close && !isdigit(static_cast<unsigned char>(text[i])) && text[i] != '-') ++i;
            if (i >= close) break;
            size_t j = i;
            while (j < close && (isdigit(static_cast<unsigned char>(text[j])) || text[j] == '-')) ++j;
            ids.push_back(std::atoi(text.substr(i, j - i).c_str()));
            i = j;
        }
        return ids;
    }

    // 读整个文件后调上面的扫描；单 prompt 路径（--tokens-json）用。
    std::vector<int> parse_tokens_json(const std::string &path) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
            std::exit(2);
        }
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::string text(static_cast<size_t>(size), '\0');
        size_t got = std::fread(text.data(), 1, text.size(), f);
        std::fclose(f);
        text.resize(got);
        return parse_tokens_from_string(text, path.c_str());
    }

    // 批量模式输入：JSONL，每行一个 {"tokens": [...]}。空行跳过（文件末尾
    // 多一个换行是常态）；非空坏行 fail fast 带行号。
    std::vector<std::vector<int>> parse_batch_jsonl(const std::string &path) {
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "error: cannot open %s\n", path.c_str());
            std::exit(2);
        }
        std::fseek(f, 0, SEEK_END);
        long size = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        std::string text(static_cast<size_t>(size), '\0');
        size_t got = std::fread(text.data(), 1, text.size(), f);
        std::fclose(f);
        text.resize(got);

        std::vector<std::vector<int>> prompts;
        size_t line_no = 0;
        size_t pos = 0;
        while (pos <= text.size()) {
            size_t nl = text.find('\n', pos);
            std::string line = text.substr(pos, (nl == std::string::npos ? text.size() : nl) - pos);
            pos = nl == std::string::npos ? text.size() + 1 : nl + 1;
            ++line_no;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            size_t first = line.find_first_not_of(" \t");
            if (first == std::string::npos) continue; // 空行跳过
            char what[160];
            std::snprintf(what, sizeof(what), "%s line %zu", path.c_str(), line_no);
            prompts.push_back(parse_tokens_from_string(line, what));
        }
        return prompts;
    }
} // namespace

int main(int argc, char **argv) {
    Args args;
    if (!parse_args(argc, argv, &args)) return 2;

    // ---- 读取配置文件（可选）----
    tinyqwen::Config config;
    if (!args.config.empty()) {
        std::string cerr;
        if (!config.load(args.config, &cerr)) {
            std::fprintf(stderr, "error: %s\n", cerr.c_str());
            return 1;
        }
        std::fprintf(stderr, "[init] config: %s (%zu keys)\n", args.config.c_str(),
                     config.size());
    }

    // ---- 解析 matvec 实现名：CLI > 配置文件 > 默认 ref ----
    // 实现名由各 kernel 文件自注册（dispatch.h），这里只按名字查表——
    // 新增变体不需要改这段代码。
    std::string impl_name = args.matvec_impl; // 非空 = CLI 显式指定
    if (impl_name.empty()) impl_name = config.get("matvec_impl", "ref");

    // ---- 加载权重文件并校验 ----
    tinyqwen::ModelFile file;
    std::string err;
    if (!file.load(args.model, &err)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (args.verbose) file.print_summary();

    // ---- 解析停止符 eos：CLI 显式值 > 模型文件头（v2）> Qwen2.5 默认 ----
    if (args.eos == -2) {
        const uint32_t hdr_eos = file.config().eos_token_id;
        args.eos = hdr_eos != 0 ? static_cast<int>(hdr_eos) : 151645;
        std::fprintf(stderr, "[init] eos: %d (auto)\n", args.eos);
    }

    // ---- 按模型 dtype 选择 matvec 实现（f32/f16 各有独立注册表）----
    // 加载模型在前、选实现在后：同一个实现名（如 "ref"/"neon_mt_kv_nt"）
    // 在两个注册表里各有一份，按文件 dtype 查对应的表——f32 模型配
    // f16 专属实现（或反过来）会在这里 fail fast，而不是静默兜底。
    const bool is_f16 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kF16);
    const bool is_i4 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kI4);
    if (is_i4) {
        if (!tinyqwen::set_matvec_i4_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr,
                         "error: unknown i4 matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_i4_impls());
            return 2;
        }
        // lm_head（tied embed，fp32）走 f32 注册表：优先同名实现，取不到退回
        // 最强 f32 实现。不设的话 lm_head 落标量 ref——每 token 544MB fp32
        // 是单项最大流量，曾占 i4 decode 90% 时间（实测 154/213 ms）。
        if (!tinyqwen::set_matvec_impl_by_name(impl_name.c_str())) {
            tinyqwen::set_matvec_impl_by_name("neon_mt_kv_nt");
        }
        std::fprintf(stderr, "[init] matvec impl: %s (i4 weights, group=%u; lm_head f32: %s)\n",
                     tinyqwen::matvec_i4_impl_name(), file.config().quant_group_size,
                     tinyqwen::matvec_impl_name());
    } else if (is_f16) {
        if (!tinyqwen::set_matvec_f16_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr,
                         "error: unknown f16 matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_f16_impls());
            return 2;
        }
        std::fprintf(stderr, "[init] matvec impl: %s (f16 weights)\n",
                     tinyqwen::matvec_f16_impl_name());
    } else {
        if (!tinyqwen::set_matvec_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr, "error: unknown matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_impls());
            return 2;
        }
        std::fprintf(stderr, "[init] matvec impl: %s\n", tinyqwen::matvec_impl_name());
    }

    // ---- 选择非 matvec 算子实现（与 dtype 无关，五算子共用一个名）----
    // 优先级 CLI > 配置 > 默认 ref。"ref" 是兜底行为（不注册、不 set），
    // 其余名字（如 neon）查五个算子注册表，任一注册即接受、未注册的算子
    // 自动兜底 ref。
    std::string ops_name = args.ops_impl;
    if (ops_name.empty()) ops_name = config.get("ops_impl", "ref");
    if (ops_name != "ref") {
        if (!tinyqwen::set_ops_impl_by_name(ops_name.c_str())) {
            std::fprintf(stderr, "error: unknown ops_impl '%s' (expected: ref / neon)\n",
                         ops_name.c_str());
            return 2;
        }
    }
    std::fprintf(stderr, "[init] ops impl: %s\n", tinyqwen::ops_impl_name());

    // profiling 按需开启：没有 --profile-out 时所有 ScopedTimer 都是空操作。
    tinyqwen::Profiler profiler(!args.profile_out.empty());
    const bool is_qwen35 = file.config().model_type == tinyqwen::ModelType::kQwen35;
    profiler.set_meta(is_qwen35 ? "qwen3.5-hybrid" : "qwen2.5-like", "cpu_ref",
                      is_f16 ? "f16w_fp32a" : "fp32");

    // ---- 建模：校验权重、分配 KV cache 和 workspace ----
    std::unique_ptr<tinyqwen::QwenModel> model;
    if (!tinyqwen::QwenModel::create(file, args.max_seq_len, profiler, &err, &model)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "[init] kv cache: %.1f MB (max_seq_len=%d)\n",
                 model->kv_cache().memory_bytes() / (1024.0 * 1024.0), args.max_seq_len);
    if (is_qwen35) {
        std::fprintf(stderr, "[init] gdn state: %.1f MB (O(1) w.r.t. seq len)\n",
                     model->gdn_state_bytes() / (1024.0 * 1024.0));
    }

    // ---- 批量模式（数据集测试）：一次进程顺序跑多条 prompt ----
    // 每条独立 reset → prefill → decode；TTFT / 逐步 decode 用 steady_clock
    // 手测，不开 op 级 profiler（跨 prompt 累积会爆内存，且本模式的契约是纯
    // 时序测量）。与 --topk/--dump-logits/--verbose/--engine/--profile-out
    // 互斥（parse_args 已拦）。
    if (!args.batch_tokens_jsonl.empty()) {
        const std::vector<std::vector<int>> prompts = parse_batch_jsonl(args.batch_tokens_jsonl);
        if (prompts.empty()) {
            std::fprintf(stderr, "error: batch file has no prompts: %s\n",
                         args.batch_tokens_jsonl.c_str());
            return 2;
        }

        using Clock = std::chrono::steady_clock;
        const auto ms_since = [](const Clock::time_point &a, const Clock::time_point &b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };

        struct PromptResult {
            int prompt_tokens = 0;
            int generated_tokens = 0;
            bool hit_eos = false;
            double ttft_ms = 0.0;
            double decode_sum_ms = 0.0;
            std::vector<double> decode_ms;
        };
        std::vector<PromptResult> results;
        results.reserve(prompts.size());
        const auto wall_start = Clock::now();

        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            const std::vector<int> &tok = prompts[pi];
            const int n = static_cast<int>(tok.size());
            if (n <= 0) {
                std::fprintf(stderr, "error: %s line %zu: empty tokens\n",
                             args.batch_tokens_jsonl.c_str(), pi + 1);
                return 2;
            }
            if (n + args.max_new_tokens > args.max_seq_len) {
                std::fprintf(stderr, "error: %s line %zu: prompt %d + decode %d > "
                                     "max_seq_len %d（加大 --max-seq-len 或裁短 prompt）\n",
                             args.batch_tokens_jsonl.c_str(), pi + 1, n,
                             args.max_new_tokens, args.max_seq_len);
                return 2;
            }

            // 关键：reset 只清 KV/GDN/token_count_，prompt_len_ 必须每条重设，
            // 否则 forward 内部的 is_prefill 判定会错乱。
            model->reset();
            model->set_prompt_len(n);

            PromptResult r;
            r.prompt_tokens = n;
            const auto t0 = Clock::now();
            int next = model->forward_prefill(tok.data(), n, nullptr, 0);
            const auto t1 = Clock::now();
            r.ttft_ms = ms_since(t0, t1);

            std::vector<int> generated;
            for (int step = 0; step < args.max_new_tokens; ++step) {
                generated.push_back(next);
                if (args.eos >= 0 && next == args.eos) {
                    r.hit_eos = true;
                    std::fprintf(stderr, "[batch] %zu/%zu hit eos %d at step %d\n",
                                 pi + 1, prompts.size(), args.eos, step);
                    break;
                }
                if (step + 1 == args.max_new_tokens) break; // 最后一个 token 不再前向
                const auto ts = Clock::now();
                next = model->forward_token(next, nullptr, 0);
                const auto te = Clock::now();
                const double step_ms = ms_since(ts, te);
                r.decode_ms.push_back(step_ms);
                r.decode_sum_ms += step_ms;
            }
            r.generated_tokens = static_cast<int>(generated.size());

            // stdout 契约：与单 prompt 模式同格式，每条一行、顺序对应 JSONL 行序。
            std::printf("generated_ids:");
            for (int id: generated) std::printf(" %d", id);
            std::printf("\n");
            std::fflush(stdout);
            std::fprintf(stderr, "[batch] %zu/%zu prompt=%d gen=%d ttft=%.2f ms\n",
                         pi + 1, prompts.size(), n, r.generated_tokens, r.ttft_ms);
            results.push_back(std::move(r));
        }
        const double wall_ms = ms_since(wall_start, Clock::now());

        // 手写结果 JSON（同 profiler 风格，不引第三方库）。C++ 只出原始数，
        // 统计口径（分桶/中位数）留给 Python 侧。
        FILE *f = std::fopen(args.batch_out.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "error: cannot write %s\n", args.batch_out.c_str());
            return 1;
        }
        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"n_prompts\": %zu,\n", results.size());
        std::fprintf(f, "  \"max_new_tokens\": %d,\n", args.max_new_tokens);
        std::fprintf(f, "  \"max_seq_len\": %d,\n", args.max_seq_len);
        std::fprintf(f, "  \"prompts\": [\n");
        for (size_t i = 0; i < results.size(); ++i) {
            const PromptResult &r = results[i];
            std::fprintf(f, "    {\"index\": %zu, \"prompt_tokens\": %d, "
                            "\"generated_tokens\": %d, \"hit_eos\": %s, "
                            "\"ttft_ms\": %.4f, \"decode_ms\": [",
                         i, r.prompt_tokens, r.generated_tokens,
                         r.hit_eos ? "true" : "false", r.ttft_ms);
            for (size_t k = 0; k < r.decode_ms.size(); ++k) {
                std::fprintf(f, "%s%.4f", k ? ", " : "", r.decode_ms[k]);
            }
            std::fprintf(f, "], \"total_ms\": %.4f}%s\n",
                         r.ttft_ms + r.decode_sum_ms,
                         i + 1 < results.size() ? "," : "");
        }
        std::fprintf(f, "  ],\n");
        std::fprintf(f, "  \"total_wall_ms\": %.4f\n", wall_ms);
        std::fprintf(f, "}\n");
        std::fclose(f);
        std::fprintf(stderr, "[batch] done: %zu prompts in %.1f ms\n", results.size(), wall_ms);
        std::fprintf(stderr, "[batch-out] %s\n", args.batch_out.c_str());
        return 0;
    }

    // ---- 选择 decode engine（GPU-resident forward，opt-in）----
    // 默认空 = CPU forward（参考路径）。--engine cuda 时把整段 forward 搬上
    // GPU：权重/激活/KV 常驻显存，单 stream 跑完，只有 token id 过 PCIe。
    // CPU 的 forward_token 原样保留作参考；engine 创建失败即报错（fail fast）。
    std::string engine_name = args.engine;
    if (engine_name.empty()) engine_name = config.get("engine", "");
    tinyqwen::GpuDecodeEngine *engine = nullptr;
    if (!engine_name.empty()) {
        if (!tinyqwen::set_gpu_decode_impl_by_name(engine_name.c_str())) {
            std::fprintf(stderr, "error: unknown engine '%s' (available: %s)\n",
                         engine_name.c_str(), tinyqwen::available_gpu_decode_impls());
            return 2;
        }
        // v1：engine 路径只支持 greedy，--topk / --dump-logits 暂不支持。
        if (args.topk > 0 || !args.dump_logits.empty()) {
            std::fprintf(stderr,
                         "error: --topk / --dump-logits 在 --engine 下暂不支持（v1）\n");
            return 2;
        }
        std::string eerr;
        if (!tinyqwen::gpu_decode_create(&file, args.max_seq_len, &eerr, &engine)) {
            std::fprintf(stderr, "error: gpu engine create failed: %s\n", eerr.c_str());
            return 1;
        }
        std::fprintf(stderr, "[init] decode engine: %s (GPU-resident forward)\n",
                     tinyqwen::gpu_decode_impl_name());
    }

    // engine 路径下由 main 驱动 profiler（forward_token 被绕过，不再自己记）。
    // pos = 当前 KV 长度 = 已处理 token 数，与 forward_token 内部口径一致。
    int engine_token_idx = 0;
    const auto engine_step = [&](int token, bool is_prefill) -> int {
        profiler.begin_token(engine_token_idx, engine_token_idx, is_prefill);
        const int nxt = tinyqwen::gpu_decode_step(engine, token);
        profiler.end_token();
        ++engine_token_idx;
        return nxt;
    };

    // prompt 的 token ids（由 Python 侧 tools/tokenize_prompt.py 生成；
    // v1 按设计不在 C++ 里做 tokenizer）。
    std::vector<int> tokens =
            args.tokens_csv.empty() ? parse_tokens_json(args.tokens_json) : parse_csv(args.tokens_csv);
    if (tokens.empty()) {
        std::fprintf(stderr, "error: empty token list\n");
        return 2;
    }
    // 融合开关：CLI > 配置文件 > 默认 true。
    const bool fuse_gate_up = args.no_fuse_gate_up ? false
                              : config.get("fuse_gate_up", "true") != "false";
    const bool fuse_qkv = args.no_fuse_qkv ? false
                           : config.get("fuse_qkv", "true") != "false";
    model->set_fuse_gate_up(fuse_gate_up);
    model->set_fuse_qkv(fuse_qkv);
    model->set_prompt_len(static_cast<int>(tokens.size()));

    FILE *logits_out = nullptr;
    if (!args.dump_logits.empty()) {
        logits_out = std::fopen(args.dump_logits.c_str(), "wb");
        if (!logits_out) {
            std::fprintf(stderr, "error: cannot open %s\n", args.dump_logits.c_str());
            return 1;
        }
    }
    // 每次 forward 追加一行 vocab 个 fp32 logits；行序与序列位置一一对应
    // （tools/align_fake_model.py 依赖这个约定）。
    const auto dump = [&]() {
        if (logits_out) {
            std::fwrite(model->last_logits(), sizeof(float), file.config().vocab_size, logits_out);
        }
    };

    // 每个 "topk" 行描述的是下一个 "gen" 行 token 的分布
    // （第一行在 prefill 结束后输出）。
    const auto print_topk = [](const tinyqwen::TopKResult &topk) {
        std::printf("topk");
        for (size_t i = 0; i < topk.indices.size(); ++i) {
            std::printf(" %d:%.6f", topk.indices[i], topk.values[i]);
        }
        std::printf("\n");
    };

    // ---- prefill 阶段 ----
    // 批量 GEMM 路径：一次处理所有 prompt token 的线性投影，显著降低延迟。
    // verbose 模式退回逐 token 以输出每步中间结果。
    int next = 0;
    tinyqwen::TopKResult topk;
    if (engine) {
        // GPU-resident engine 没有批量 prefill 入口，只能逐 token 喂进去
        for (size_t i = 0; i < tokens.size(); ++i) {
            next = engine_step(tokens[i], true);
            if (args.verbose) {
                std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                             tokens[i], next);
            }
        }
    } else if (args.verbose) {
        // 逐 token：可打印每步详情
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool last_prefill = i + 1 == tokens.size();
            const bool need_topk = args.topk > 0 && last_prefill;
            next = model->forward_token(tokens[i], need_topk ? &topk : nullptr, args.topk);
            dump();
            std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                         tokens[i], next);
        }
    } else {
        // 批量 GEMM prefill。整批作为一条 prefill 记录计时：TTFT 需要它
        // （此前这条路径完全没进 profiler，profile 里 prompt_tokens=0、
        // first_token_ms=0、total_ms 也漏掉 prefill 耗时）。批量 GEMM 的算子
        // 结构与逐 token 不同，不做 op 级拆分。
        const bool need_topk = args.topk > 0;
        profiler.begin_token(0, 0, /*is_prefill=*/true);
        next = model->forward_prefill(tokens.data(), static_cast<int>(tokens.size()),
                                      need_topk ? &topk : nullptr, args.topk);
        profiler.end_token();
    }
    std::fprintf(stderr, "[prefill] %zu tokens done\n", tokens.size());
    if (args.topk > 0) print_topk(topk); // 第一个生成 token g0 的分数分布

    // ---- decode 阶段 ----
    // 循环：把上一步的输出当作下一步的输入，每次生成一个新 token，
    // 直到凑够 max_new_tokens 或遇到停止符 eos。
    std::vector<int> generated;
    for (int step = 0; step < args.max_new_tokens; ++step) {
        generated.push_back(next);
        std::printf("gen %d %d\n", step, next);
        if (args.eos >= 0 && next == args.eos) {
            std::fprintf(stderr, "[decode] hit eos %d at step %d\n", args.eos, step);
            break;
        }
        if (step + 1 == args.max_new_tokens) break; // 最后一个 token 不用再前向
        if (engine) {
            next = engine_step(next, false); // GPU-resident forward
        } else {
            next = model->forward_token(next, args.topk > 0 ? &topk : nullptr, args.topk);
            dump();
        }
        if (args.topk > 0) print_topk(topk); // 下一个 gen token 的分布
    }

    std::printf("generated_ids:");
    for (int id: generated) std::printf(" %d", id);
    std::printf("\n");

    // 显式声明真实计数，覆盖"按记录数推导"的口径偏差：
    // 批量 prefill 只有一条记录（≠ prompt 长度）；decode 步数 = 生成数 - 1。
    profiler.set_counts(tokens.size(), generated.size());

    // 释放 GPU engine（若有）。CPU 路径无操作。
    if (engine) tinyqwen::gpu_decode_destroy(engine);

    if (!args.profile_out.empty()) {
        std::string werr;
        if (!profiler.write_json(args.profile_out, &werr)) {
            std::fprintf(stderr, "error: %s\n", werr.c_str());
            return 1;
        }
        std::fprintf(stderr, "[profile] %s\n", args.profile_out.c_str());
    }
    if (logits_out) std::fclose(logits_out);
    return 0;
}
