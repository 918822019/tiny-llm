// tinyqwen CLI：加载 .tqwen 模型，跑 token-by-token greedy decode。
//
//   tinyqwen --model model.tqwen \
//            --tokens-json prompt_tokens.json \
//            --max-new-tokens 16 \
//            --profile-out profile.json
//
// token ids 来自 Python（tools/tokenize_prompt.py）：v1 的 C++ 侧
// 刻意不内置 tokenizer。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

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
  int eos = 151645;  // Qwen2.5 的 im_end；传 -1 禁用
  bool verbose = false;
};

void usage(const char* prog) {
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
               "  --verbose               model summary + per-token details\n",
               prog);
}

bool parse_args(int argc, char** argv, Args* out) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    // 取 flag 的值；缺值直接报错退出。
    const auto value = [&](const char* flag) -> std::string {
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
    else if (a == "--verbose") out->verbose = true;
    else if (a == "--help" || a == "-h") { usage(argv[0]); std::exit(0); }
    else {
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
std::vector<int> parse_csv(const std::string& s) {
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
std::vector<int> parse_tokens_json(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
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

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) return 2;

  tinyqwen::ModelFile file;
  std::string err;
  if (!file.load(args.model, &err)) {
    std::fprintf(stderr, "error: %s\n", err.c_str());
    return 1;
  }
  if (args.verbose) file.print_summary();

  // profiling 按需开启：没有 --profile-out 时所有 ScopedTimer 都是空操作。
  tinyqwen::Profiler profiler(!args.profile_out.empty());
  profiler.set_meta("qwen2.5-0.5b-like", "cpu_ref", "fp32");

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
  model->set_prompt_len(static_cast<int>(tokens.size()));

  FILE* logits_out = nullptr;
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
  const auto print_topk = [](const tinyqwen::TopKResult& topk) {
    std::printf("topk");
    for (size_t i = 0; i < topk.indices.size(); ++i) {
      std::printf(" %d:%.6f", topk.indices[i], topk.values[i]);
    }
    std::printf("\n");
  };

  // ---- prefill（token-by-token）----
  // prefill 结束后，`next` 就是第一个生成 token
  //（最后一个 prompt 位置 logits 的 argmax）。
  int next = 0;
  tinyqwen::TopKResult topk;
  for (size_t i = 0; i < tokens.size(); ++i) {
    const bool last_prefill = i + 1 == tokens.size();
    next = model->forward_token(
        tokens[i], (args.topk > 0 && last_prefill) || args.verbose ? &topk : nullptr,
        args.topk);
    dump();
    if (args.verbose) {
      std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                   tokens[i], next);
    }
  }
  std::fprintf(stderr, "[prefill] %zu tokens done\n", tokens.size());
  if (args.topk > 0) print_topk(topk);  // g0 的分布

  // ---- decode ----
  std::vector<int> generated;
  for (int step = 0; step < args.max_new_tokens; ++step) {
    generated.push_back(next);
    std::printf("gen %d %d\n", step, next);
    if (args.eos >= 0 && next == args.eos) {
      std::fprintf(stderr, "[decode] hit eos %d at step %d\n", args.eos, step);
      break;
    }
    if (step + 1 == args.max_new_tokens) break;  // 最后一个 token 不用再前向
    next = model->forward_token(next, args.topk > 0 ? &topk : nullptr, args.topk);
    dump();
    if (args.topk > 0) print_topk(topk);  // 下一个 gen token 的分布
  }

  std::printf("generated_ids:");
  for (int id : generated) std::printf(" %d", id);
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
