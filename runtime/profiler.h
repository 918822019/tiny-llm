#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tinyqwen {

// Lightweight per-token / per-op profiler with a JSON sink.
//
// Usage:
//   Profiler profiler(true);
//   profiler.set_meta("qwen2.5-0.5b", "cpu_ref", "fp32");
//   ...
//   profiler.begin_token(i, pos, /*is_prefill=*/i < n_prompt);
//   { ScopedTimer t(profiler, "layer_0.q_proj"); ... }
//   profiler.end_token();
//   profiler.write_json("profile.json", &err);
//
// Notes:
//   - disabled profiler: all calls are near-zero cost;
//   - ops recorded outside begin_token()/end_token() are dropped;
//   - clock is std::chrono::steady_clock (fine on Android NDK);
//   - JSON is emitted by hand: no third-party dependency.
// Schema: docs/profiling_schema.md.
class Profiler {
 public:
  struct OpRecord {
    std::string name;
    double latency_ms = 0.0;
  };
  struct TokenRecord {
    int index = 0;
    int pos = 0;
    bool is_prefill = false;
    double latency_ms = 0.0;
    std::vector<OpRecord> ops;
  };
  struct OpStat {
    uint64_t calls = 0;
    double total_ms = 0.0;
  };

  explicit Profiler(bool enabled = true);

  bool enabled() const { return enabled_; }
  void set_meta(const std::string& model, const std::string& backend,
                const std::string& precision);

  void begin_token(int index, int pos, bool is_prefill = false);
  void end_token();
  void enter(const char* name);
  void leave(const char* name);

  const std::vector<TokenRecord>& tokens() const { return tokens_; }
  const std::map<std::string, OpStat>& op_totals() const { return op_totals_; }

  bool write_json(const std::string& path, std::string* err) const;

 private:
  struct Frame {
    std::string name;
    std::chrono::steady_clock::time_point start;
  };
  using Clock = std::chrono::steady_clock;

  static double ms_since(const Clock::time_point& start, const Clock::time_point& end);

  bool enabled_;
  std::string model_ = "unknown";
  std::string backend_ = "cpu_ref";
  std::string precision_ = "fp32";

  bool in_token_ = false;
  Clock::time_point token_start_;
  TokenRecord current_;
  std::vector<Frame> stack_;

  std::vector<TokenRecord> tokens_;
  std::map<std::string, OpStat> op_totals_;
};

class ScopedTimer {
 public:
  ScopedTimer(Profiler& profiler, const char* name) : profiler_(profiler), name_(name) {
    profiler_.enter(name_);
  }
  ~ScopedTimer() { profiler_.leave(name_); }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  Profiler& profiler_;
  const char* name_;
};

}  // namespace tinyqwen
