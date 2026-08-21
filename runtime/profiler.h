#pragma once

// ============================================================================
// 文件: profiler.h
// 作用: Profiler —— 记录"每一步花了多长时间"，最后输出 JSON 供分析
//
// 为什么需要 Profiler?
//   想优化性能，先得知道时间花在哪。profiler 把每次 forward 拆成一个个 op
//   （如 "layer_0.q_proj"），分别计时，这样就能看出瓶颈在哪个算子。
//
// 核心设计:
//   1. RAII 计时器 (ScopedTimer): 构造时开始计时，析构时停止。
//      好处是只要对象离开作用域就一定"停止"，不怕中途 return/抛异常时忘记停。
//   2. 按 token 组织: 每个 token 的处理拆成多个 op，按 token 聚合。
//   3. 零开销开关: profiler 关闭时所有调用接近零开销（不计时就不产生代价）。
//   4. JSON 手写输出: 不依赖第三方库，输出 schema 见 docs/profiling_schema.md。
//
// 前置阅读: docs/infra_primer.md 第 7 节（RAII 与时钟）
// ============================================================================

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tinyqwen {
    // -------------------------------------------------------------------------
    // Profiler: 轻量的 per-token / per-op 性能分析器
    //
    // 用法示例:
    //   Profiler profiler(true);
    //   profiler.set_meta("qwen2.5-0.5b", "cpu_ref", "fp32");
    //
    //   // 每个 token 开始
    //   profiler.begin_token(i, pos, /*is_prefill=*/i < n_prompt);
    //   {
    //       // 给一段代码计时
    //       ScopedTimer t(profiler, "layer_0.q_proj");
    //       // ... 计算 ...
    //   }
    //   profiler.end_token();  // 一个 token 结束
    //
    //   // 最终输出
    //   std::string err;
    //   profiler.write_json("profile.json", &err);
    //
    // 说明:
    //   - profiler 关闭时所有调用接近零开销
    //   - begin_token()/end_token() 之外记录的 op 会被丢弃
    //   - 时钟用 steady_clock: 单调递增、不受系统时间被改影响，适合测耗时
    //     （wall clock 会因为对时/NTP 跳变，不能用来测时长）
    // -------------------------------------------------------------------------
    class Profiler {
    public:
        // ---------------------------------------------------------------------
        // OpRecord: 单个 op 的计时记录
        //
        // 记录一个 op 的名字和它花费的时间（毫秒）。
        // 例如: name = "layer_0.q_proj", latency_ms = 0.042
        // ---------------------------------------------------------------------
        struct OpRecord {
            std::string name;       // op 名称，如 "layer_0.q_proj"
            double latency_ms = 0.0; // 耗时（毫秒）
        };

        // ---------------------------------------------------------------------
        // TokenRecord: 一个 token 的完整记录
        //
        // 包含该 token 的处理耗时以及它包含的所有 op 的计时记录。
        // ---------------------------------------------------------------------
        struct TokenRecord {
            int index = 0;                 // 这是第几个被处理的 token（0-based 计数）
            int pos = 0;                   // 它在序列里的位置（用于 prefill 时区分具体位置）
            bool is_prefill = false;       // 属于 prefill 还是 decode 阶段
            double latency_ms = 0.0;       // 总耗时（毫秒），所有 op 的延迟之和
            std::vector<OpRecord> ops;     // 该 token 内所有 op 的计时记录
        };

        // ---------------------------------------------------------------------
        // OpStat: 某个 op 跨所有 token 的累计统计
        //
        // 用于按 op 聚合分析，看哪个 op 最耗时。
        // ---------------------------------------------------------------------
        struct OpStat {
            uint64_t calls = 0;    // 被调用的总次数
            double total_ms = 0.0; // 累计总耗时（毫秒）
        };

        // ---------------------------------------------------------------------
        // 构造函数: 创建 profiler
        //
        // 参数:
        //   enabled: 是否启用计时。false 时所有调用近似零开销。
        // ---------------------------------------------------------------------
        explicit Profiler(bool enabled = true);

        // 是否启用
        bool enabled() const { return enabled_; }

        // ---------------------------------------------------------------------
        // set_meta: 设置元信息
        //
        // 参数:
        //   model:     模型名称，如 "qwen2.5-0.5b"
        //   backend:   后端名称，如 "cpu_ref" / "cuda"
        //   precision: 精度，如 "fp32" / "fp16" / "int4"
        //
        // 说明:
        //   这些元信息会写入 JSON 输出，方便后续分析和对比不同配置的性能。
        // ---------------------------------------------------------------------
        void set_meta(const std::string &model, const std::string &backend,
                      const std::string &precision);

        // ---------------------------------------------------------------------
        // begin_token: 开始记录一个 token 的处理
        //
        // 参数:
        //   index:      这是第几个被处理的 token（0-based 全局计数）
        //   pos:        它在序列里的位置
        //   is_prefill: 是否属于 prefill 阶段（true）还是 decode 阶段（false）
        //
        // 说明:
        //   必须在 end_token() 之前调用。begin/end 之间记录的所有 op 才有效。
        //   begin/end 之外调用的 enter/leave 记录的 op 会被丢弃。
        // ---------------------------------------------------------------------
        void begin_token(int index, int pos, bool is_prefill = false);

        // ---------------------------------------------------------------------
        // end_token: 结束当前 token 的记录
        //
        // 说明:
        //   将当前 token 的记录保存到 tokens_ 列表，并聚合到 op_totals_。
        //   之后再次 begin_token 开始下一个 token。
        // ---------------------------------------------------------------------
        void end_token();

        // ---------------------------------------------------------------------
        // enter: 一个 op 开始计时
        //
        // 参数:
        //   name: op 名称，如 "layer_0.q_proj"
        //
        // 说明:
        //   一般不直接调用，而是通过 ScopedTimer 的 RAII 机制自动管理。
        //   支持嵌套: 内部的 enter/leave 会在外层 op 的子区间内累积。
        // ---------------------------------------------------------------------
        void enter(const char *name);

        // ---------------------------------------------------------------------
        // leave: 一个 op 结束计时
        //
        // 参数:
        //   name: op 名称，必须与对应的 enter 匹配
        //
        // 说明:
        //   计算从 enter 到 leave 的 wall clock 耗时。
        // ---------------------------------------------------------------------
        void leave(const char *name);

        // ---------------------------------------------------------------------
        // set_counts: 显式声明 prompt 长度和生成 token 数
        //
        // 参数:
        //   prompt_tokens:    prompt 的 token 总数
        //   generated_tokens: 生成的 token 总数
        //
        // 说明:
        //   write_json 优先使用这些值。为什么不能只靠 token 记录数推导:
        //     - 批量 prefill 下整个 prompt 只有一条记录（记录数=1 ≠ prompt 长度）
        //     - decode 步数 = 生成数 - 1（最后一个 token 不需要再 forward）
        // ---------------------------------------------------------------------
        void set_counts(uint64_t prompt_tokens, uint64_t generated_tokens);

        // ---- 数据访问 ----

        // 所有已完成的 token 记录（只读）
        const std::vector<TokenRecord> &tokens() const { return tokens_; }

        // 跨 token 的 op 聚合统计（只读）
        const std::map<std::string, OpStat> &op_totals() const { return op_totals_; }

        // ---------------------------------------------------------------------
        // write_json: 将 profiling 结果写入 JSON 文件
        //
        // 参数:
        //   path: 输出文件路径
        //   err:  输出参数，失败时写入错误原因
        //
        // 返回值:
        //   成功返回 true，失败返回 false
        //
        // 说明:
        //   JSON 手写输出，不依赖第三方库。输出 schema 见 docs/profiling_schema.md。
        // ---------------------------------------------------------------------
        bool write_json(const std::string &path, std::string *err) const;

    private:
        // ---------------------------------------------------------------------
        // Frame: 作用域栈上的一帧
        //
        // 记录一个 op 的名字和开始时刻，用于支持嵌套计时。
        // enter() 时压栈，leave() 时弹栈并计算耗时。
        // ---------------------------------------------------------------------
        struct Frame {
            std::string name;                           // op 名称
            std::chrono::steady_clock::time_point start; // 开始时刻
        };

        // 时钟类型别名: steady_clock 单调递增，适合测时间间隔
        using Clock = std::chrono::steady_clock;

        // ---------------------------------------------------------------------
        // ms_since: 计算两个时间点之间的毫秒差
        //
        // 参数:
        //   start: 开始时刻
        //   end:   结束时刻
        //
        // 返回值:
        //   时间差（毫秒），double 精度
        // ---------------------------------------------------------------------
        static double ms_since(const Clock::time_point &start, const Clock::time_point &end);

        // ---- 配置和状态 ----

        bool enabled_;                    // 是否启用计时
        std::string model_ = "unknown";   // 模型名称
        std::string backend_ = "cpu_ref"; // 后端名称
        std::string precision_ = "fp32";  // 精度

        bool in_token_ = false;           // 当前是否在一个 token 的记录范围内
        Clock::time_point token_start_;   // 当前 token 的开始时刻
        TokenRecord current_;             // 正在累积的 token 记录
        std::vector<Frame> stack_;        // enter/leave 的作用域栈（支持嵌套）

        // ---- 持久化数据 ----

        std::vector<TokenRecord> tokens_;              // 所有已完成的 token 记录
        std::map<std::string, OpStat> op_totals_;      // 跨 token 的 op 聚合统计

        // 显式计数（通过 set_counts 设置）
        // 未设置时（nullopt），write_json 退回按记录推导
        std::optional<uint64_t> prompt_tokens_;
        std::optional<uint64_t> generated_tokens_;
    };

    // -------------------------------------------------------------------------
    // ScopedTimer: RAII 计时器
    //
    // 用法:
    //   {
    //       ScopedTimer t(profiler, "layer_0.q_proj");
    //       // ... 计算 ...
    //   } // 离开作用域时自动停止计时
    //
    // 设计原理:
    //   RAII（Resource Acquisition Is Initialization）:
    //     构造时获取资源（开始计时），析构时释放资源（停止计时）。
    //     好处: 只要对象离开作用域就一定"停止"，不怕中途 return/抛异常时忘记停。
    //
    // 生命周期:
    //   - 禁止拷贝（避免重复计时/提前停止）
    //   - profiler 引用必须比 ScopedTimer 活得更久
    // -------------------------------------------------------------------------
    class ScopedTimer {
    public:
        // ---------------------------------------------------------------------
        // 构造函数: 开始计时
        //
        // 参数:
        //   profiler: 关联的 Profiler 对象
        //   name:     op 名称，如 "layer_0.q_proj"
        // ---------------------------------------------------------------------
        ScopedTimer(Profiler &profiler, const char *name) : profiler_(profiler), name_(name) {
            profiler_.enter(name_); // 记录开始时刻
        }

        // ---------------------------------------------------------------------
        // 析构函数: 停止计时
        // ---------------------------------------------------------------------
        ~ScopedTimer() { profiler_.leave(name_); } // 记录结束时刻并计算耗时

        // 禁止拷贝（RAII 对象通常不应拷贝，避免重复释放）
        ScopedTimer(const ScopedTimer &) = delete;
        ScopedTimer &operator=(const ScopedTimer &) = delete;

    private:
        Profiler &profiler_;  // 关联的 profiler（引用，不拥有）
        const char *name_;    // op 名称
    };
} // namespace tinyqwen