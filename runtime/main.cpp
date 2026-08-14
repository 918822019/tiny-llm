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
        std::string profile_out;
        std::string dump_logits;
        int max_new_tokens = 16;
        int max_seq_len = 1024;
        int topk = 0;
        int eos = 151645; // Qwen2.5 的 im_end；传 -1 禁用
        bool verbose = false;
        bool no_fuse_gate_up = false;
        bool no_fuse_qkv = false;
        std::string config; // 配置文件路径（可选）
        std::string matvec_impl; // matvec 实现；空 = 未指定，交给配置/默认值
        std::string ops_impl; // 非 matvec 算子实现；空 = 未指定，交给配置/默认值
    };

    void usage(const char *prog) {
        std::fprintf(stderr,
                     "usage: %s --model <model.tqwen> [options]\n"
                     "  --tokens CSV            comma separated token ids\n"
                     "  --tokens-json PATH      JSON with a \"tokens\" array (tokenize_prompt.py)\n"
                     "  --max-new-tokens N      default 16\n"
                     "  --max-seq-len N         KV capacity, default 1024\n"
                     "  --topk K                emit top-k logits lines (each describes the next\n"
                     "                          gen token; 0 = off)\n"
                     "  --dump-logits PATH      dump full logits (fp32 binary) after every forward\n"
                     "  --profile-out PATH      write profiler JSON\n"
                     "  --eos ID                stop token, default 151645, -1 disables\n"
                     "  --config PATH           key=value config file (CLI flags override it)\n"
                     "  --matvec-impl NAME      matvec kernel: ref (default; neon later)\n"
                     "  --ops-impl NAME         non-matvec ops (rmsnorm/rope/attention/swiglu/\n"
                     "                          argmax): ref (default) / neon\n"
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
            else if (a == "--max-new-tokens") out->max_new_tokens = std::atoi(value("--max-new-tokens").c_str());
            else if (a == "--max-seq-len") out->max_seq_len = std::atoi(value("--max-seq-len").c_str());
            else if (a == "--topk") out->topk = std::atoi(value("--topk").c_str());
            else if (a == "--profile-out") out->profile_out = value("--profile-out");
            else if (a == "--dump-logits") out->dump_logits = value("--dump-logits");
            else if (a == "--eos") out->eos = std::atoi(value("--eos").c_str());
            else if (a == "--config") out->config = value("--config");
            else if (a == "--matvec-impl") out->matvec_impl = value("--matvec-impl");
            else if (a == "--ops-impl") out->ops_impl = value("--ops-impl");
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
        // --tokens 与 --tokens-json 必须且只能提供一个。
        if (out->tokens_csv.empty() == out->tokens_json.empty()) {
            std::fprintf(stderr, "error: provide exactly one of --tokens / --tokens-json\n");
            return false;
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

    // 从 JSON 中做最小化的 "tokens" 整数数组提取；不引 JSON 依赖。
    // 约定见 tools/tokenize_prompt.py 的输出格式。
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

        size_t key = text.find("\"tokens\"");
        if (key == std::string::npos) {
            std::fprintf(stderr, "error: no \"tokens\" field in %s\n", path.c_str());
            std::exit(2);
        }
        size_t open = text.find('[', key);
        size_t close = text.find(']', open);
        if (open == std::string::npos || close == std::string::npos) {
            std::fprintf(stderr, "error: malformed tokens array in %s\n", path.c_str());
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

    // ---- 按模型 dtype 选择 matvec 实现（f32/f16 各有独立注册表）----
    // 加载模型在前、选实现在后：同一个实现名（如 "ref"/"neon_mt_kv_nt"）
    // 在两个注册表里各有一份，按文件 dtype 查对应的表——f32 模型配
    // f16 专属实现（或反过来）会在这里 fail fast，而不是静默兜底。
    const bool is_f16 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kF16);
    if (is_f16) {
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
    profiler.set_meta("qwen2.5-0.5b-like", "cpu_ref", is_f16 ? "f16w_fp32a" : "fp32");

    // ---- 建模：校验权重、分配 KV cache 和 workspace ----
    std::unique_ptr<tinyqwen::QwenModel> model;
    if (!tinyqwen::QwenModel::create(file, args.max_seq_len, profiler, &err, &model)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "[init] kv cache: %.1f MB (max_seq_len=%d)\n",
                 model->kv_cache().memory_bytes() / (1024.0 * 1024.0), args.max_seq_len);

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

    // ---- prefill 阶段（token-by-token）----
    // 把 prompt 的每个 token 依次喂进模型。每步都填 KV cache 并产生 logits，
    // 但我们只关心最后一步——它的 argmax 就是第一个要生成的 token。
    // prefill 结束后，`next` 就是第一个生成 token。
    int next = 0;
    tinyqwen::TopKResult topk;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const bool last_prefill = i + 1 == tokens.size();
        // 是否要让 forward 顺带返回 top-k：verbose 时需要；或者开了 --topk
        // 且这是最后一个 prompt token（要打印第一个生成 token 的分布）。
        const bool need_topk = args.verbose || (args.topk > 0 && last_prefill);
        next = model->forward_token(tokens[i], need_topk ? &topk : nullptr, args.topk);
        dump();
        if (args.verbose) {
            std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                         tokens[i], next);
        }
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
        next = model->forward_token(next, args.topk > 0 ? &topk : nullptr, args.topk);
        dump();
        if (args.topk > 0) print_topk(topk); // 下一个 gen token 的分布
    }

    std::printf("generated_ids:");
    for (int id: generated) std::printf(" %d", id);
    std::printf("\n");

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
