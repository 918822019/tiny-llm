// Profiler implementation. Recording model:
//   - begin_token()/end_token() delimit one forward pass;
//   - enter()/leave() (via ScopedTimer) push/pop a small stack, so nested
//     scopes are tolerated, though the v1 forward uses flat, disjoint scopes;
//   - every recorded op is appended to the current token in execution order
//     AND aggregated into op_totals, so both per-token traces and global
//     op shares come from one pass over the data.
// The JSON layout written below is specified in docs/profiling_schema.md.

#include "profiler.h"

#include <cstdio>

namespace tinyqwen {

Profiler::Profiler(bool enabled) : enabled_(enabled) {}

void Profiler::set_meta(const std::string& model, const std::string& backend,
                        const std::string& precision) {
  model_ = model;
  backend_ = backend;
  precision_ = precision;
}

double Profiler::ms_since(const Clock::time_point& start, const Clock::time_point& end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void Profiler::begin_token(int index, int pos, bool is_prefill) {
  if (!enabled_) return;
  in_token_ = true;
  token_start_ = Clock::now();
  current_ = TokenRecord{};
  current_.index = index;
  current_.pos = pos;
  current_.is_prefill = is_prefill;
  stack_.clear();
}

void Profiler::end_token() {
  if (!enabled_ || !in_token_) return;
  current_.latency_ms = ms_since(token_start_, Clock::now());
  tokens_.push_back(std::move(current_));
  in_token_ = false;
  stack_.clear();
}

// Ops outside a token record are dropped on purpose (model loading etc.
// should not pollute per-token traces).
void Profiler::enter(const char* name) {
  if (!enabled_ || !in_token_) return;
  stack_.push_back(Frame{std::string(name), Clock::now()});
}

void Profiler::leave(const char* name) {
  if (!enabled_ || !in_token_ || stack_.empty()) return;
  Frame frame = std::move(stack_.back());
  stack_.pop_back();
  if (frame.name != name) {
    std::fprintf(stderr, "tinyqwen: profiler scope mismatch: enter '%s' leave '%s'\n",
                 frame.name.c_str(), name);
  }
  const double ms = ms_since(frame.start, Clock::now());
  current_.ops.push_back(OpRecord{std::move(frame.name), ms});
  OpStat& stat = op_totals_[current_.ops.back().name];
  stat.calls += 1;
  stat.total_ms += ms;
}

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

}  // namespace

bool Profiler::write_json(const std::string& path, std::string* err) const {
  // Aggregate summary.
  double total_ms = 0.0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  uint64_t n_prefill = 0, n_decode = 0;
  for (const TokenRecord& t : tokens_) {
    total_ms += t.latency_ms;
    if (t.is_prefill) {
      prefill_ms += t.latency_ms;
      ++n_prefill;
    } else {
      decode_ms += t.latency_ms;
      ++n_decode;
    }
  }

  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    if (err) *err = "cannot open for write: " + path;
    return false;
  }

  // Hand-written JSON (no third-party dependency). Field semantics:
  //   first_token_ms = sum of prefill token latencies (~TTFT for
  //                    token-by-token prefill), decode_avg_ms over the rest.
  std::fprintf(f, "{\n");
  std::fprintf(f, "  \"model\": \"%s\",\n", json_escape(model_).c_str());
  std::fprintf(f, "  \"backend\": \"%s\",\n", json_escape(backend_).c_str());
  std::fprintf(f, "  \"precision\": \"%s\",\n", json_escape(precision_).c_str());
  std::fprintf(f, "  \"prompt_tokens\": %llu,\n", (unsigned long long)n_prefill);
  std::fprintf(f, "  \"generated_tokens\": %llu,\n", (unsigned long long)n_decode);
  std::fprintf(f, "  \"total_ms\": %.4f,\n", total_ms);
  std::fprintf(f, "  \"first_token_ms\": %.4f,\n", prefill_ms);
  std::fprintf(f, "  \"decode_avg_ms\": %.4f,\n", n_decode ? decode_ms / n_decode : 0.0);
  std::fprintf(f, "  \"tokens\": [\n");
  for (size_t i = 0; i < tokens_.size(); ++i) {
    const TokenRecord& t = tokens_[i];
    std::fprintf(f, "    {\"index\": %d, \"pos\": %d, \"is_prefill\": %s, "
                    "\"latency_ms\": %.4f, \"ops\": {",
                 t.index, t.pos, t.is_prefill ? "true" : "false", t.latency_ms);
    for (size_t j = 0; j < t.ops.size(); ++j) {
      std::fprintf(f, "%s\"%s\": %.4f", j ? ", " : "", json_escape(t.ops[j].name).c_str(),
                   t.ops[j].latency_ms);
    }
    std::fprintf(f, "}}%s\n", i + 1 < tokens_.size() ? "," : "");
  }
  std::fprintf(f, "  ],\n");
  std::fprintf(f, "  \"op_totals\": {\n");
  size_t i = 0;
  for (const auto& [name, stat] : op_totals_) {
    std::fprintf(f, "    \"%s\": {\"calls\": %llu, \"total_ms\": %.4f}%s\n",
                 json_escape(name).c_str(), (unsigned long long)stat.calls, stat.total_ms,
                 ++i < op_totals_.size() ? "," : "");
  }
  std::fprintf(f, "  }\n");
  std::fprintf(f, "}\n");
  std::fclose(f);
  return true;
}

}  // namespace tinyqwen
