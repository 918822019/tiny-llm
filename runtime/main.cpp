// ============================================================================
// main.cpp — tinyqwen 命令行入口
// ============================================================================
// 本文件是整个 tiny-llm 推理引擎的命令行入口，负责将"加载模型 → 前向推理
// → 输出结果"这一完整流程串联起来。支持两种推理模式：
//
// 1. 单条推理模式：
//      加载一个 .tqwen 模型文件，对一组 prompt token 做 prefill（预填充）
//      然后进入 decode（逐 token 生成）循环，输出生成的 token 序列。
//
// 2. 批量测试模式（--batch-tokens-jsonl）：
//      一次进程加载模型，对 JSONL 中每条 prompt 独立做 reset → prefill → decode，
//      输出每条的时间测量结果（TTFT、每步 decode 耗时）到 JSON 文件。
//
// 推理分为两个阶段（概念见 docs/infra_primer.md 第 2 节）：
//   - prefill（预填充）：把输入的每个 prompt token 依次喂进模型，填满 KV cache，
//     最后一步产出第一个生成 token；
//   - decode（解码）：之后每步把上一步生成的 token 再喂回去，生成下一个，循环。
//
// 典型用法：
//   tinyqwen --model model.tqwen \
//            --tokens-json prompt_tokens.json \
//            --max-new-tokens 16 \
//            --profile-out profile.json
//
// token ids 来自 Python（tools/tokenize_prompt.py）：v1 的 C++ 侧刻意不内置
// tokenizer（分词与研究主线无关，复用现成工具即可）。
// ============================================================================

#include <chrono>       // 高精度计时（std::chrono::steady_clock）
#include <cstdio>        // 标准输入输出（fprintf, printf, fflush）
#include <cstdlib>       // 标准库（atoi, exit）
#include <cstring>       // C 字符串操作
#include <memory>        // 智能指针（std::unique_ptr, std::make_unique）
#include <string>        // C++ 字符串
#include <vector>        // 动态数组

#include "backend_cpu.h"
#include "config.h"
#include "dispatch.h"
#include "model_loader.h"
#include "profiler.h"
#include "qwen_model.h"

#ifdef TINYQWEN_HAS_CUDA
#include "backend_cuda.h"
#endif

namespace {
    // 命令行参数结构体：集中存储所有 CLI 选项
    struct Args {
        std::string model;             // 模型文件路径（.tqwen），必选
        std::string tokens_csv;        // CSV 格式 token 列表（如 "1,2,3"）
        std::string tokens_json;       // JSON 格式 token 文件
        std::string batch_tokens_jsonl; // 批量输入 JSONL 文件路径
        std::string batch_out;          // 批量输出 JSON 文件路径
        std::string profile_out;        // 性能剖析输出 JSON 路径
        std::string dump_logits;        // 输出 logits 的二进制文件路径
        int max_new_tokens = 16;        // 最大生成 token 数，默认 16
        int max_seq_len = 1024;         // KV cache 容量（最大序列长度），默认 1024
        int topk = 0;                   // 输出 top-k logits 的行数，0 表示关闭
        int eos = -2;                   // 停止符：-2=自动，-1=禁用，其余值=显式指定
        bool verbose = false;           // 是否输出详细调试信息
        bool no_fuse_gate_up = false;   // 禁用 gate/up 投影融合
        bool no_fuse_qkv = false;       // 禁用 Q/K/V 投影融合
        std::string config;             // 配置文件路径（可选）
        std::string matvec_impl;        // matvec 实现选择；空 = 未指定
        std::string ops_impl;           // 非 matvec 算子实现；空 = 未指定
        std::string engine;             // decode engine；空 = CPU forward（默认）
        std::string backend;            // 计算后端；空 = CPU（默认）
    };

    // 打印命令行帮助信息
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
                     "  --backend NAME          compute backend: '' = CPU (default) / cuda\n"
                     "                          (per-operator CUDA; requires CUDA build)\n"
                     "  --verbose               model summary + per-token details\n",
                     prog);
    }

    // 解析命令行参数
    // 返回值：成功返回 true，失败返回 false
    bool parse_args(int argc, char **argv, Args *out) {
        // 从索引 1 开始遍历（跳过程序名 argv[0]）
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            // value lambda：获取当前 flag 的下一个参数作为值，缺值则报错退出
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
            else if (a == "--backend") out->backend = value("--backend");
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
        // 三种输入模式必须且只能提供一个
        const bool has_batch = !out->batch_tokens_jsonl.empty();
        const int modes = (!out->tokens_csv.empty()) + (!out->tokens_json.empty()) + has_batch;
        if (modes != 1) {
            std::fprintf(stderr, "error: provide exactly one of "
                                 "--tokens / --tokens-json / --batch-tokens-jsonl\n");
            return false;
        }
        if (has_batch) {
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

    // 解析 CSV 格式的 token ID 列表（如 "1,2,3"）
    // 容忍空格，遇到非法字符直接报错
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

    // 从 JSON 文本中最小化提取 "tokens" 整数数组，不引入 JSON 解析库依赖
    // 参数 what 用于报错定位（文件路径或 "<path> line N"）
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

    // 从 JSON 文件读取 token 列表（单 prompt 路径用）
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

    // 解析批量输入 JSONL 文件，每行一个 {"tokens": [...]}
    // 空行跳过，非空坏行 fail fast 带行号
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
    std::string impl_name = args.matvec_impl;
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

    // ---- 按模型 dtype 选择 matvec 实现（f32/f16/i4 各有独立注册表）----
    const bool is_f16 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kF16);
    const bool is_i4 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kI4);
    if (is_i4) {
        if (!tinyqwen::set_matvec_i4_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr,
                         "error: unknown i4 matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_i4_impls());
            return 2;
        }
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

    // profiling 按需开启：没有 --profile-out 时所有 ScopedTimer 都是空操作
    tinyqwen::Profiler profiler(!args.profile_out.empty());
    const bool is_qwen35 = file.config().model_type == tinyqwen::ModelType::kQwen35;
    profiler.set_meta(is_qwen35 ? "qwen3.5-hybrid" : "qwen2.5-like", "cpu_ref",
                      is_f16 ? "f16w_fp32a" : "fp32");

    // ---- 创建后端（默认 CPU，可选 CUDA）----
    std::unique_ptr<tinyqwen::IBackend> backend;
    if (args.backend == "cuda") {
#ifdef TINYQWEN_HAS_CUDA
        backend = tinyqwen::create_cuda_backend();
        std::fprintf(stderr, "[init] backend: CUDA\n");
#else
        std::fprintf(stderr, "error: --backend cuda requires CUDA build\n");
        return 1;
#endif
    } else {
        backend = tinyqwen::create_cpu_backend();
        if (!args.backend.empty()) {
            std::fprintf(stderr, "[init] backend: CPU (default)\n");
        }
    }

    // ---- 建模：校验权重、分配 KV cache 和 workspace ----
    std::unique_ptr<tinyqwen::QwenModel> model;
    if (!tinyqwen::QwenModel::create(file, args.max_seq_len, profiler, &err, &model, std::move(backend))) {
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
                if (step + 1 == args.max_new_tokens) break;
                const auto ts = Clock::now();
                next = model->forward_token(next, nullptr, 0);
                const auto te = Clock::now();
                const double step_ms = ms_since(ts, te);
                r.decode_ms.push_back(step_ms);
                r.decode_sum_ms += step_ms;
            }
            r.generated_tokens = static_cast<int>(generated.size());

            std::printf("generated_ids:");
            for (int id: generated) std::printf(" %d", id);
            std::printf("\n");
            std::fflush(stdout);
            std::fprintf(stderr, "[batch] %zu/%zu prompt=%d gen=%d ttft=%.2f ms\n",
                         pi + 1, prompts.size(), n, r.generated_tokens, r.ttft_ms);
            results.push_back(std::move(r));
        }
        const double wall_ms = ms_since(wall_start, Clock::now());

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
    std::string engine_name = args.engine;
    if (engine_name.empty()) engine_name = config.get("engine", "");
    tinyqwen::GpuDecodeEngine *engine = nullptr;
    if (!engine_name.empty()) {
        if (!tinyqwen::set_gpu_decode_impl_by_name(engine_name.c_str())) {
            std::fprintf(stderr, "error: unknown engine '%s' (available: %s)\n",
                         engine_name.c_str(), tinyqwen::available_gpu_decode_impls());
            return 2;
        }
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

    // engine 路径下由 main 驱动 profiler（forward_token 被绕过，不再自己记）
    int engine_token_idx = 0;
    const auto engine_step = [&](int token, bool is_prefill) -> int {
        profiler.begin_token(engine_token_idx, engine_token_idx, is_prefill);
        const int nxt = tinyqwen::gpu_decode_step(engine, token);
        profiler.end_token();
        ++engine_token_idx;
        return nxt;
    };

    // prompt 的 token ids（由 Python 侧 tools/tokenize_prompt.py 生成）
    std::vector<int> tokens =
            args.tokens_csv.empty() ? parse_tokens_json(args.tokens_json) : parse_csv(args.tokens_csv);
    if (tokens.empty()) {
        std::fprintf(stderr, "error: empty token list\n");
        return 2;
    }
    // 融合开关：CLI > 配置文件 > 默认 true
    const bool fuse_gate_up = args.no_fuse_gate_up ? false
                              : config.get("fuse_gate_up", "true") != "false";
    const bool fuse_qkv = args.no_fuse_qkv ? false
                           : config.get("fuse_qkv", "true") != "false";
    model->set_fuse_gate_up(fuse_gate_up);
    model->set_fuse_qkv(fuse_qkv);
    model->set_prompt_len(static_cast<int>(tokens.size()));

    // 打开 logits 输出文件（如果指定了 --dump-logits）
    FILE *logits_out = nullptr;
    if (!args.dump_logits.empty()) {
        logits_out = std::fopen(args.dump_logits.c_str(), "wb");
        if (!logits_out) {
            std::fprintf(stderr, "error: cannot open %s\n", args.dump_logits.c_str());
            return 1;
        }
    }
    // dump lambda：每次 forward 追加一行 vocab 个 fp32 logits
    const auto dump = [&]() {
        if (logits_out) {
            std::fwrite(model->last_logits(), sizeof(float), file.config().vocab_size, logits_out);
        }
    };

    // 打印 top-k 结果
    const auto print_topk = [](const tinyqwen::TopKResult &topk) {
        std::printf("topk");
        for (size_t i = 0; i < topk.indices.size(); ++i) {
            std::printf(" %d:%.6f", topk.indices[i], topk.values[i]);
        }
        std::printf("\n");
    };

    // ---- prefill 阶段 ----
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
        // 逐 token prefill：可打印每步详情
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool last_prefill = i + 1 == tokens.size();
            const bool need_topk = args.topk > 0 && last_prefill;
            next = model->forward_token(tokens[i], need_topk ? &topk : nullptr, args.topk);
            dump();
            std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                         tokens[i], next);
        }
    } else {
        // 批量 GEMM prefill：整批作为一条 prefill 记录计时
        const bool need_topk = args.topk > 0;
        profiler.begin_token(0, 0, /*is_prefill=*/true);
        next = model->forward_prefill(tokens.data(), static_cast<int>(tokens.size()),
                                      need_topk ? &topk : nullptr, args.topk);
        profiler.end_token();
    }
    std::fprintf(stderr, "[prefill] %zu tokens done\n", tokens.size());
    if (args.topk > 0) print_topk(topk);

    // ---- decode 阶段 ----
    std::vector<int> generated;
    for (int step = 0; step < args.max_new_tokens; ++step) {
        generated.push_back(next);
        std::printf("gen %d %d\n", step, next);
        if (args.eos >= 0 && next == args.eos) {
            std::fprintf(stderr, "[decode] hit eos %d at step %d\n", args.eos, step);
            break;
        }
        if (step + 1 == args.max_new_tokens) break;
        if (engine) {
            next = engine_step(next, false);
        } else {
            next = model->forward_token(next, args.topk > 0 ? &topk : nullptr, args.topk);
            dump();
        }
        if (args.topk > 0) print_topk(topk);
    }

    std::printf("generated_ids:");
    for (int id: generated) std::printf(" %d", id);
    std::printf("\n");

    profiler.set_counts(tokens.size(), generated.size());

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