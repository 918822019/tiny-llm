// ============================================================================
// profiler.cpp — 性能剖析器实现
// ============================================================================
// 本文件实现 Profiler 类，用于记录模型推理过程中的时间消耗。
//
// 记录模型：
//   - begin_token()/end_token() 框定一次 forward 的起止时间；
//   - enter()/leave()（经 ScopedTimer RAII 包装）压入/弹出一个小型作用域栈，
//     因此允许嵌套作用域，不过 v1 的 forward 用的是扁平、互不重叠的作用域；
//   - 每个记录的 op 既按执行顺序追加到当前 token，也累加进 op_totals，
//     于是一次数据就能同时得到 per-token trace 和全局 op 占比。
//
// 数据输出：
//   set_meta() 设置模型元信息（模型名、后端、精度）；
//   set_counts() 显式设置 prompt_tokens 和 generated_tokens 计数；
//   write_json() 将收集的数据序列化为 JSON 文件。
//
// 下面写出的 JSON 布局见 docs/profiling_schema.md。
// ============================================================================

#include "profiler.h"

#include <cstdio>  // 标准输入输出（fopen, fclose, fprintf, snprintf）

namespace tinyqwen {
    // =========================================================================
    // Profiler::Profiler() — 构造函数
    // =========================================================================
    // 参数：
    //   enabled — 是否启用性能剖析。false 时所有记录操作都是空操作（零开销）。
    // 说明：这样设计让 profiling 在非 profiling 模式下几乎零开销——所有
    //       ScopedTimer 的 enter/leave 调用在 !enabled_ 时直接返回。
    Profiler::Profiler(bool enabled) : enabled_(enabled) {
    }

    // =========================================================================
    // Profiler::set_meta() — 设置模型元信息
    // =========================================================================
    // 参数：
    //   model     — 模型名称（如 "qwen3.5-hybrid" 或 "qwen2.5-like"）
    //   backend   — 后端名称（如 "cpu_ref"）
    //   precision — 精度描述（如 "f16w_fp32a" 或 "fp32"）
    // 说明：这些信息会写入 JSON 输出的顶层字段，方便后续分析时区分不同配置。
    void Profiler::set_meta(const std::string &model, const std::string &backend,
                            const std::string &precision) {
        model_ = model;
        backend_ = backend;
        precision_ = precision;
    }

    // =========================================================================
    // Profiler::ms_since() — 计算两个时间点之间的毫秒差
    // =========================================================================
    // 参数：
    //   start — 起始时间点
    //   end   — 结束时间点
    // 返回值：时间差（毫秒），double 精度
    // 说明：使用 std::chrono::duration<double, std::milli> 进行亚毫秒级测量。
    double Profiler::ms_since(const Clock::time_point &start, const Clock::time_point &end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    // =========================================================================
    // Profiler::begin_token() — 开始记录一个 token 的 forward
    // =========================================================================
    // 参数：
    //   index      — token 索引（序列中的第几个 token，从 0 开始）
    //   pos        — KV cache 中的位置（与 index 可能不同，批量 prefill 时）
    //   is_prefill — 是否为 prefill 阶段（true=prefill, false=decode）
    // 说明：记录当前 token 的开始时间，重置 op 栈。如果 profiler 未启用，
    //       直接返回（零开销）。
    void Profiler::begin_token(int index, int pos, bool is_prefill) {
        if (!enabled_) return; // 未启用，直接返回
        in_token_ = true;
        token_start_ = Clock::now(); // 记录 token 开始时间
        current_ = TokenRecord{};     // 重置当前 token 记录
        current_.index = index;
        current_.pos = pos;
        current_.is_prefill = is_prefill;
        stack_.clear(); // 清空作用域栈
    }

    // =========================================================================
    // Profiler::end_token() — 结束当前 token 的记录
    // =========================================================================
    // 说明：计算 token 总延迟，将当前记录追加到 tokens_ 列表中。
    //       如果 profiler 未启用或不在 token 记录中，直接返回。
    void Profiler::end_token() {
        if (!enabled_ || !in_token_) return;
        // 计算从 begin_token 到现在的总耗时
        current_.latency_ms = ms_since(token_start_, Clock::now());
        tokens_.push_back(std::move(current_)); // 移动语义，避免拷贝
        in_token_ = false;
        stack_.clear();
    }

    // =========================================================================
    // Profiler::enter() — 进入一个 profiling 作用域
    // =========================================================================
    // 参数：
    //   name — 作用域名称（如 "layer_0.qkv_proj"）
    // 说明：将当前时间压入作用域栈。token 记录之外的 op 有意丢弃
    //       （模型加载等耗时不应污染 per-token trace）。
    void Profiler::enter(const char *name) {
        if (!enabled_ || !in_token_) return; // 未启用或不在 token 记录中
        stack_.push_back(Frame{std::string(name), Clock::now()});
    }

    // =========================================================================
    // Profiler::leave() — 离开一个 profiling 作用域
    // =========================================================================
    // 参数：
    //   name — 作用域名称（应与 enter 时一致）
    // 说明：弹出栈顶，计算耗时，追加到当前 token 的 op 列表和全局 op 统计。
    //       如果 name 与栈顶名称不一致，打印警告（不中断，仅提示）。
    void Profiler::leave(const char *name) {
        if (!enabled_ || !in_token_ || stack_.empty()) return;
        Frame frame = std::move(stack_.back());
        stack_.pop_back();
        // 名称匹配检查：不一致时打印警告
        if (frame.name != name) {
            std::fprintf(stderr, "tinyqwen: profiler scope mismatch: enter '%s' leave '%s'\n",
                         frame.name.c_str(), name);
        }
        // 计算该作用域耗时（毫秒）
        const double ms = ms_since(frame.start, Clock::now());
        // 追加到当前 token 的 op 列表
        current_.ops.push_back(OpRecord{std::move(frame.name), ms});
        // 累加到全局 op 统计
        OpStat &stat = op_totals_[current_.ops.back().name];
        stat.calls += 1;      // 调用次数 +1
        stat.total_ms += ms;  // 累计耗时
    }

    // =========================================================================
    // Profiler::set_counts() — 显式设置 token 计数
    // =========================================================================
    // 参数：
    //   prompt_tokens    — prompt 阶段的 token 数量
    //   generated_tokens — 生成的 token 数量
    // 说明：覆盖"按记录数推导"的口径偏差。批量 prefill 只有一条记录
    //       （不等于 prompt 长度），decode 步数 = 生成数 - 1，因此需要
    //       显式声明真实计数。
    void Profiler::set_counts(uint64_t prompt_tokens, uint64_t generated_tokens) {
        prompt_tokens_ = prompt_tokens;
        generated_tokens_ = generated_tokens;
    }

    namespace {
        // =====================================================================
        // json_escape() — 对字符串进行 JSON 转义
        // =====================================================================
        // 参数：
        //   s — 原始字符串
        // 返回值：转义后的 JSON 安全字符串
        // 说明：处理双引号、反斜杠、换行、制表符、回车以及控制字符（< 0x20）。
        //       不引入第三方 JSON 库，手写最小化转义。
        std::string json_escape(const std::string &s) {
            std::string out;
            out.reserve(s.size() + 8); // 预分配空间，减少 reallocation
            for (char c: s) {
                switch (c) {
                    case '"': out += "\\\""; break;   // 双引号转义
                    case '\\': out += "\\\\"; break;  // 反斜杠转义
                    case '\n': out += "\\n"; break;   // 换行符
                    case '\t': out += "\\t"; break;   // 制表符
                    case '\r': out += "\\r"; break;   // 回车符
                    default:
                        if (static_cast<unsigned char>(c) < 0x20) {
                            // 控制字符：转义为 \u00xx 格式
                            char buf[8];
                            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                            out += buf;
                        } else {
                            out += c; // 普通字符直接添加
                        }
                }
            }
            return out;
        }
    } // namespace

    // =========================================================================
    // Profiler::write_json() — 将 profiling 数据写入 JSON 文件
    // =========================================================================
    // 参数：
    //   path — 输出文件路径
    //   err  — 输出参数，出错时写入错误信息
    // 返回值：成功返回 true，失败返回 false
    // 说明：手写 JSON（不引第三方库）。输出结构包含：
    //   - 模型元信息（model, backend, precision）
    //   - 摘要统计（prompt_tokens, generated_tokens, total_ms, first_token_ms,
    //     decode_avg_ms）
    //   - 每个 token 的详细记录（tokens 数组）
    //   - 全局 op 统计（op_totals）
    // 字段语义：
    //   first_token_ms = TTFT（Time To First Token）：prefill 记录耗时之和。
    //                    批量 prefill 下就是整批记录的耗时（真 TTFT）；
    //                    token-by-token 路径下是各 prefill token 之和。
    //   decode_avg_ms  只统计 decode token 的平均耗时。
    bool Profiler::write_json(const std::string &path, std::string *err) const {
        // ---- 汇总统计 ----
        double total_ms = 0.0;
        double prefill_ms = 0.0;
        double decode_ms = 0.0;
        uint64_t n_prefill = 0, n_decode = 0;
        for (const TokenRecord &t: tokens_) {
            total_ms += t.latency_ms;
            if (t.is_prefill) {
                prefill_ms += t.latency_ms;
                ++n_prefill;
            } else {
                decode_ms += t.latency_ms;
                ++n_decode;
            }
        }

        // 显式计数优先（set_counts）：批量 prefill / decode 步数与生成数的
        // 口径差异见 profiler.h 的说明
        const uint64_t out_prompt = prompt_tokens_.value_or(n_prefill);
        const uint64_t out_generated = generated_tokens_.value_or(n_decode);

        // 打开输出文件
        FILE *f = std::fopen(path.c_str(), "w");
        if (!f) {
            if (err) *err = "cannot open for write: " + path;
            return false;
        }

        // ---- 手写 JSON 输出 ----
        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"model\": \"%s\",\n", json_escape(model_).c_str());
        std::fprintf(f, "  \"backend\": \"%s\",\n", json_escape(backend_).c_str());
        std::fprintf(f, "  \"precision\": \"%s\",\n", json_escape(precision_).c_str());
        std::fprintf(f, "  \"prompt_tokens\": %llu,\n", (unsigned long long) out_prompt);
        std::fprintf(f, "  \"generated_tokens\": %llu,\n", (unsigned long long) out_generated);
        std::fprintf(f, "  \"total_ms\": %.4f,\n", total_ms);
        std::fprintf(f, "  \"first_token_ms\": %.4f,\n", prefill_ms);
        std::fprintf(f, "  \"decode_avg_ms\": %.4f,\n", n_decode ? decode_ms / n_decode : 0.0);
        // 每个 token 的详细记录
        std::fprintf(f, "  \"tokens\": [\n");
        for (size_t i = 0; i < tokens_.size(); ++i) {
            const TokenRecord &t = tokens_[i];
            std::fprintf(f, "    {\"index\": %d, \"pos\": %d, \"is_prefill\": %s, "
                         "\"latency_ms\": %.4f, \"ops\": {",
                         t.index, t.pos, t.is_prefill ? "true" : "false", t.latency_ms);
            for (size_t j = 0; j < t.ops.size(); ++j) {
                std::fprintf(f, "%s\"%s\": %.4f", j ? ", " : "", json_escape(t.ops[j].name).c_str(),
                             t.ops[j].latency_ms);
            }
            std::fprintf(f, "}}%s\n", i + 1 < tokens_.size() ? "," : ""); // 最后一条不加逗号
        }
        std::fprintf(f, "  ],\n");
        // 全局 op 统计
        std::fprintf(f, "  \"op_totals\": {\n");
        size_t i = 0;
        for (const auto &[name, stat]: op_totals_) {
            std::fprintf(f, "    \"%s\": {\"calls\": %llu, \"total_ms\": %.4f}%s\n",
                         json_escape(name).c_str(), (unsigned long long) stat.calls, stat.total_ms,
                         ++i < op_totals_.size() ? "," : ""); // 最后一条不加逗号
        }
        std::fprintf(f, "  }\n");
        std::fprintf(f, "}\n");
        std::fclose(f);
        return true;
    }
} // namespace tinyqwen