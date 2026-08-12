#pragma once

// Profiler：记录"每一步花了多长时间"，最后输出 JSON 供分析。
// 先读 docs/infra_primer.md 第 7 节（RAII 与时钟）。
//
// 为什么需要它？想优化性能，先得知道时间花在哪。profiler 把每次 forward
// 拆成一个个 op（如 "layer_0.q_proj"），分别计时，这样就能看出瓶颈在哪个算子。

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace tinyqwen {

// 轻量的 per-token / per-op profiler，输出 JSON。
//
// 用法：
//   Profiler profiler(true);
//   profiler.set_meta("qwen2.5-0.5b", "cpu_ref", "fp32");
//   ...
//   profiler.begin_token(i, pos, /*is_prefill=*/i < n_prompt);  // 一个 token 开始
//   { ScopedTimer t(profiler, "layer_0.q_proj"); ... }          // 给一段代码计时
//   profiler.end_token();                                       // 一个 token 结束
//   profiler.write_json("profile.json", &err);                  // 落盘
//
// 说明：
//   - profiler 关闭时所有调用接近零开销（不计时就不产生代价）；
//   - begin_token()/end_token() 之外记录的 op 会被丢弃；
//   - 时钟用 steady_clock：单调递增、不受系统时间被改影响，适合测耗时
//     （wall clock 会因为对时/NTP 跳变，不能用来测时长）；
//   - JSON 手写输出，不依赖第三方库。
// schema 见 docs/profiling_schema.md。
class Profiler {
 public:
  // 一个 op 的计时记录。
  struct OpRecord {
    std::string name;
    double latency_ms = 0.0;
  };
  // 一个 token 的完整记录：耗时 + 它包含的所有 op。
  struct TokenRecord {
    int index = 0;          // 这是第几个被处理的 token
    int pos = 0;            // 它在序列里的位置
    bool is_prefill = false;  // 属于 prefill 还是 decode 阶段
    double latency_ms = 0.0;
    std::vector<OpRecord> ops;
  };
  // 某个 op 跨所有 token 的累计统计。
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
  void enter(const char* name);  // 一个 op 开始（一般不直接调，用 ScopedTimer）
  void leave(const char* name);  // 一个 op 结束

  const std::vector<TokenRecord>& tokens() const { return tokens_; }
  const std::map<std::string, OpStat>& op_totals() const { return op_totals_; }

  bool write_json(const std::string& path, std::string* err) const;

 private:
  // 作用域栈上的一帧：记录 op 名字和开始时刻。
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

  bool in_token_ = false;            // 当前是否在一个 token 的记录范围内
  Clock::time_point token_start_;    // 当前 token 的开始时刻
  TokenRecord current_;              // 正在累积的 token 记录
  std::vector<Frame> stack_;         // enter/leave 的作用域栈（支持嵌套）

  std::vector<TokenRecord> tokens_;         // 所有已完成的 token 记录
  std::map<std::string, OpStat> op_totals_; // 跨 token 的 op 聚合
};

// RAII 计时器：构造时开始计时，析构时停止计时。
// 好处：只要对象离开作用域就一定"停止"，不怕中途 return/抛异常时忘记停。
// （RAII 是什么见 primer 第 7 节。）
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
