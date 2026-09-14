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
#include <cmath>         // std::exp（PPL）
#include <cstdio>        // 标准输入输出（fprintf, printf, fflush）
#include <cstdlib>       // 标准库（atoi, exit）
#include <cstring>       // C 字符串操作
#include <limits>        // std::numeric_limits（GPU 内存预算溢出校验）
#include <memory>        // 智能指针（std::unique_ptr, std::make_unique）
#include <string>        // C++ 字符串
#include <vector>        // 动态数组

#if defined(__APPLE__)
#include <mach/mach.h>   // host_statistics64（查可用内存，内存预算校验用）
#include <sys/sysctl.h>  // sysctlbyname（查物理内存）
#include <sys/types.h>
#elif defined(__linux__)
#include <unistd.h>      // sysconf（查物理内存）
#endif

#include "backend_cpu.h"
#include "backend_vulkan.h"
#include "config.h"
#include "dflash_model.h"
#include "eagle3_model.h"
#include "dispatch.h"
#include "metal_prefill.h"
#include "model_loader.h"
#include "profiler.h"
#include "qwen_model.h"
#include "speculative_decoder.h"

#ifdef TINYQWEN_HAS_CUDA
#include "backend_cuda.h"
#endif

namespace {
    // 物理内存总量（字节）。查询失败返回 0，调用方据此跳过校验——
    // 拿不到真实值时宁可放行也不要凭空造一个上限。
    uint64_t physical_memory_bytes() {
#if defined(__APPLE__)
        int64_t mem = 0;
        size_t len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0 && mem > 0) {
            return static_cast<uint64_t>(mem);
        }
        return 0;
#elif defined(__linux__)
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long page_size = sysconf(_SC_PAGESIZE);
        if (pages > 0 && page_size > 0) {
            return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
        }
        return 0;
#else
        return 0;
#endif
    }

    // 当前**可用**内存（字节）。macOS 使用 free + inactive + purgeable；
    // Linux 优先读取内核给出的 MemAvailable（无需 swap 即可分配的估算值）。
    // 为什么不能拿物理总量当上限：wired（内核与不可换出部分）+ 其他进程已占掉
    // 大半，实测 16 GB 机器上 wired 就有 8 GB、可用只剩 1.5 GB。按物理总量校验
    // 会宽松 7×，放行后照样把机器推进换页——而 swap 是**写**操作，消耗 SSD 寿命。
    uint64_t available_memory_bytes() {
#if defined(__APPLE__)
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t vm;
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64,
                              reinterpret_cast<host_info64_t>(&vm), &count) != KERN_SUCCESS) {
            return 0;
        }
        const uint64_t page = static_cast<uint64_t>(vm_page_size);
        return (static_cast<uint64_t>(vm.free_count) +
                static_cast<uint64_t>(vm.inactive_count) +
                static_cast<uint64_t>(vm.purgeable_count)) * page;
#elif defined(__linux__)
        // _SC_AVPHYS_PAGES 在 Android/Bionic 上只对应完全空闲页，不包含可立即
        // 回收的 page cache。模型 mmap/load 前用它会把 5~6 GB 的真实可用内存
        // 误报成不足 1 GB。Linux 3.14+ 的 MemAvailable 正是“不触发 swap 还能
        // 分配多少”的内核估算，与这里的 fail-fast 语义一致。
        if (FILE *meminfo = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            unsigned long long available_kib = 0;
            while (std::fgets(line, sizeof(line), meminfo)) {
                if (std::sscanf(line, "MemAvailable: %llu kB", &available_kib) == 1)
                    break;
            }
            std::fclose(meminfo);
            if (available_kib > 0) {
                return static_cast<uint64_t>(available_kib) * 1024u;
            }
        }
        // 兼容没有 MemAvailable 的旧内核。
        const long avail = sysconf(_SC_AVPHYS_PAGES);
        const long page_size = sysconf(_SC_PAGESIZE);
        if (avail > 0 && page_size > 0) {
            return static_cast<uint64_t>(avail) * static_cast<uint64_t>(page_size);
        }
        return 0;
#else
        return 0;
#endif
    }

    // Vulkan FP16 路径的 header 校验与 lm_head 尺寸计算。DFlash 整段执行器会
    // 同时复制 target + draft 权重，但 target/draft 共用唯一一份 target lm_head。
    bool f16_lm_head_bytes(const std::string &path, uint64_t *out,
                           std::string *err) {
        if (!out) return false;
        *out = 0;
        FILE *file = std::fopen(path.c_str(), "rb");
        if (!file) {
            if (err) *err = "cannot open model header for Vulkan memory estimate: " + path;
            return false;
        }
        tinyqwen::TinyHeader header{};
        const bool read_ok = std::fread(&header, sizeof(header), 1, file) == 1;
        std::fclose(file);
        if (!read_ok || std::memcmp(header.magic, tinyqwen::kMagic,
                                    sizeof(tinyqwen::kMagic)) != 0) {
            if (err) *err = "invalid model header for Vulkan memory estimate: " + path;
            return false;
        }
        if (header.dtype != static_cast<uint32_t>(tinyqwen::Dtype::kF16)) {
            if (err) *err = "Android Vulkan backend currently requires an FP16 target model";
            return false;
        }
        const uint64_t hidden = header.hidden_size;
        const uint64_t vocab = header.vocab_size;
        if (!hidden || !vocab ||
            hidden > std::numeric_limits<uint64_t>::max() / sizeof(uint16_t) / vocab) {
            if (err) *err = "invalid target lm_head shape in Vulkan memory estimate";
            return false;
        }
        *out = hidden * vocab * sizeof(uint16_t);
        return true;
    }

    // 命令行参数结构体：集中存储所有 CLI 选项
    struct Args {
        std::string model;             // 模型文件路径（.tqwen），必选
        std::string draft_model;       // 草稿模型；非空时启用投机解码
        std::string dflash_model;      // DFlash/DFlare 草稿 checkpoint
        std::string eagle3_model;      // EAGLE3 target-feature 草稿 checkpoint
        int speculative_tokens = 4;    // AR: proposal 数；DFlash/EAGLE3: 含 root 的块宽
        std::string speculative_stats_out; // 投机统计 JSON
        bool no_eagle3_batch_verify = false; // EAGLE3 逐 token 验证，仅用于消融
        std::string draft_matvec_impl; // 草稿模型独立 dtype 注册表的 kernel
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
        bool no_batch_prefill = false;  // 禁用 Qwen3.5 批量 prefill（A/B 对照用）
        bool kv_fp16 = false;           // KV cache 用 fp16 存储（内存/带宽减半，opt-in）
        bool moe_ssd = false;           // MoE：路由专家走 SSD 卸载（ExpertStore pread+LRU）
        int moe_cache_slots = 4;       // MoE 专家 LRU 缓存槽数（0 = 全 miss）
        int moe_cache_mb = 0;           // MoE 专家缓存字节预算（MB）；>0 时覆盖 slots
        bool moe_prefetch = false;      // MoE 异步预取（B-2：pread 与计算重叠）
        int moe_prefetch_slots = 8;     // 预取缓冲槽数；应 ≥ experts_per_tok
        int moe_expert_threads = 0;     // MoE 专家级并行线程数（0 = 串行原路径）
        bool force_over_mem_budget = false;  // 强制跳过内存预算校验（会换页，不推荐）
        bool ppl = false;               // PPL 模式：teacher-forcing 困惑度（不生成）
        std::string ppl_jsonl;          // PPL 多序列输入 JSONL（每行 {"tokens": [...]}）
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
                     "  --draft-model PATH      enable exact greedy speculative decoding\n"
                     "  --dflash-model PATH     enable DFlare + Markov block decoding\n"
                     "  --eagle3-model PATH     enable EAGLE3 greedy chain decoding\n"
                     "  --speculative-tokens K AR proposal count; EAGLE3/DFlash verify\n"
                     "                          width including pending root (default 4)\n"
                     "  --speculative-stats-out PATH\n"
                     "                          write acceptance/rollback/timing JSON\n"
                     "  --no-eagle3-batch-verify\n"
                     "                          verify EAGLE3 target inputs one by one\n"
                     "                          (correctness/performance ablation only)\n"
                     "  --draft-matvec-impl NAME\n"
                     "                          matvec kernel for the draft model dtype\n"
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
                     "  --kv-f16                store KV cache in fp16 (halves KV memory + attn\n"
                     "                          read bandwidth; store fp16, compute fp32)\n"
                     "  --moe-ssd               MoE: route experts via SSD offload (ExpertStore\n"
                     "                          pread + LRU) instead of resident pointers\n"
                     "  --moe-expert-cache-slots N\n"
                     "  --moe-expert-cache-mb N     专家缓存字节预算（MB），覆盖 slots；\n"
                     "                              超限 fail-fast 而非静默换页\n"
                     "  --moe-prefetch              异步预取专家（B-2：pread 与计算重叠）\n"
                      "  --moe-prefetch-slots N      预取缓冲槽数（默认 8，应 ≥ top-k）\n"
                      "  --moe-expert-threads N      专家级并行线程数（0=串行原路径；top-k 个\n"
                      "                              专家同时算，绕开 LRU 直读到 staging）\n"
                     "  --force-over-memory-budget  强制跳过内存预算校验（会换页、伤 SSD）\n"
                     "                          MoE expert LRU slots (0 = every expert re-pread;\n"
                     "                          default 4)\n"
                     "  --engine NAME           engine: '' = CPU forward (default) / cuda (decode) / metal (prefill)\n"
                     "                          (GPU-resident whole-forward; requires CUDA build)\n"
                     "  --backend NAME          compute backend: '' = CPU (default) / cuda / vulkan\n"
                     "                          (Vulkan: Android FP16 matrix ops, persistent weights)\n"
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
            else if (a == "--draft-model") out->draft_model = value("--draft-model");
            else if (a == "--dflash-model") out->dflash_model = value("--dflash-model");
            else if (a == "--eagle3-model") out->eagle3_model = value("--eagle3-model");
            else if (a == "--speculative-tokens") out->speculative_tokens = std::atoi(value("--speculative-tokens").c_str());
            else if (a == "--speculative-stats-out") out->speculative_stats_out = value("--speculative-stats-out");
            else if (a == "--no-eagle3-batch-verify") out->no_eagle3_batch_verify = true;
            else if (a == "--draft-matvec-impl") out->draft_matvec_impl = value("--draft-matvec-impl");
            else if (a == "--tokens") out->tokens_csv = value("--tokens");
            else if (a == "--tokens-json") out->tokens_json = value("--tokens-json");
            else if (a == "--batch-tokens-jsonl") out->batch_tokens_jsonl = value("--batch-tokens-jsonl");
            else if (a == "--ppl") out->ppl = true;
            else if (a == "--ppl-jsonl") out->ppl_jsonl = value("--ppl-jsonl");
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
            else if (a == "--no-batch-prefill") out->no_batch_prefill = true;
            else if (a == "--kv-f16") out->kv_fp16 = true;
            else if (a == "--moe-ssd") out->moe_ssd = true;
            else if (a == "--moe-expert-cache-slots") out->moe_cache_slots = std::atoi(value("--moe-expert-cache-slots").c_str());
            else if (a == "--moe-expert-cache-mb") out->moe_cache_mb = std::atoi(value("--moe-expert-cache-mb").c_str());
            else if (a == "--moe-prefetch") out->moe_prefetch = true;
            else if (a == "--moe-prefetch-slots") out->moe_prefetch_slots = std::atoi(value("--moe-prefetch-slots").c_str());
            else if (a == "--moe-expert-threads") out->moe_expert_threads = std::atoi(value("--moe-expert-threads").c_str());
            else if (a == "--force-over-memory-budget") out->force_over_mem_budget = true;
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
        if (out->draft_model.empty() && out->dflash_model.empty() &&
            out->eagle3_model.empty() &&
            (!out->speculative_stats_out.empty() || !out->draft_matvec_impl.empty() ||
             out->no_eagle3_batch_verify)) {
            std::fprintf(stderr,
                         "error: speculative options require a draft model option\n");
            return false;
        }
        const int draft_modes = !out->draft_model.empty() + !out->dflash_model.empty() +
                                !out->eagle3_model.empty();
        if (draft_modes > 1) {
            std::fprintf(stderr, "error: choose only one draft model option\n");
            return false;
        }
        // 三种输入模式必须且只能提供一个
        const bool has_batch = !out->batch_tokens_jsonl.empty();
        const int modes = (!out->tokens_csv.empty()) + (!out->tokens_json.empty()) + has_batch;
        const bool ppl_jsonl_only = out->ppl && !out->ppl_jsonl.empty();
        if (modes != 1 && !(ppl_jsonl_only && modes == 0)) {
            std::fprintf(stderr, "error: provide exactly one of "
                                 "--tokens / --tokens-json / --batch-tokens-jsonl"
                                 "（--ppl --ppl-jsonl 可单独使用）\n");
            return false;
        }
        if (!out->draft_model.empty()) {
            if (out->speculative_tokens <= 0) {
                std::fprintf(stderr, "error: --speculative-tokens must be positive\n");
                return false;
            }
            if (has_batch || out->ppl || out->topk > 0 || !out->dump_logits.empty() ||
                !out->profile_out.empty() || out->engine == "cuda") {
                std::fprintf(stderr,
                             "error: --draft-model currently cannot be combined with batch/PPL/"
                             "top-k/logit dump/profile/CUDA decode engine\n");
                return false;
            }
        }
        if (!out->dflash_model.empty()) {
            if (!out->draft_matvec_impl.empty()) {
                std::fprintf(stderr,
                             "error: --draft-matvec-impl applies only to --draft-model\n");
                return false;
            }
            if (out->speculative_tokens < 2) {
                std::fprintf(stderr, "error: DFlash --speculative-tokens must be >= 2\n");
                return false;
            }
            if (has_batch || out->ppl || out->topk > 0 || !out->dump_logits.empty() ||
                !out->profile_out.empty() || !out->engine.empty() || out->backend == "cuda") {
                std::fprintf(stderr,
                             "error: --dflash-model requires single-prompt CPU/Vulkan inference\n");
                return false;
            }
        }
        if (!out->eagle3_model.empty()) {
            if (!out->draft_matvec_impl.empty()) {
                std::fprintf(stderr,
                             "error: --draft-matvec-impl applies only to --draft-model\n");
                return false;
            }
            if (out->speculative_tokens < 2) {
                std::fprintf(stderr, "error: EAGLE3 --speculative-tokens must be >= 2\n");
                return false;
            }
            if (has_batch || out->ppl || out->topk > 0 || !out->dump_logits.empty() ||
                !out->profile_out.empty() || !out->engine.empty() || out->backend == "cuda") {
                std::fprintf(stderr,
                             "error: --eagle3-model requires single-prompt CPU/Vulkan inference\n");
                return false;
            }
        }
        if (out->no_eagle3_batch_verify && out->eagle3_model.empty()) {
            std::fprintf(stderr,
                         "error: --no-eagle3-batch-verify requires --eagle3-model\n");
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
    if (args.backend == "vulkan" && !tinyqwen::vulkan_backend_available()) {
        std::fprintf(stderr,
                     "error: --backend vulkan requires an Android Vulkan build\n");
        return 1;
    }

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

    // ---- 内存预算预检（必须在 load 之前）----
    // 为什么在 load() 之前：load() 会把权重真正读进 RAM。校验放在之后，等它失败时
    // 文件已经进内存了——换页已经发生，fail-fast 失去意义。estimate_resident_bytes
    // 只读 header + tensor 表、不碰数据区，所以零字节权重进内存就能算准需求。
    // 覆盖**所有**模型，不只 MoE：非 MoE（如 4B i4 4.91 GB）此前完全没有校验，
    // 加载时直接换页且零警告。
    // 判据用 available_memory_bytes()（free+inactive+purgeable）而非物理总量：
    // wired + 其他进程已占掉大半，按物理总量校验会宽松数倍，放行后照样换页。
    // swap 是写操作、消耗 SSD 寿命，所以宁可 fail-fast。留 10% 余量给 KV cache /
    // workspace / 栈等尚未计入的开销。
    {
        std::string err;
        uint64_t need = 0;
        if (!tinyqwen::ModelFile::estimate_resident_bytes(args.model, args.moe_ssd,
                                                           &need, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        const uint64_t target_need = need;
        if (!args.draft_model.empty()) {
            uint64_t draft_need = 0;
            if (!tinyqwen::ModelFile::estimate_resident_bytes(args.draft_model, false,
                                                               &draft_need, &err)) {
                std::fprintf(stderr, "error: draft model: %s\n", err.c_str());
                return 1;
            }
            need += draft_need;
        }
        uint64_t dflash_need = 0;
        if (!args.dflash_model.empty()) {
            if (!tinyqwen::ModelFile::estimate_resident_bytes(args.dflash_model, false,
                                                               &dflash_need, &err)) {
                std::fprintf(stderr, "error: DFlash model: %s\n", err.c_str());
                return 1;
            }
            need += dflash_need;
        }
        if (!args.eagle3_model.empty()) {
            uint64_t eagle3_need = 0;
            if (!tinyqwen::ModelFile::estimate_resident_bytes(args.eagle3_model, false,
                                                               &eagle3_need, &err)) {
                std::fprintf(stderr, "error: EAGLE3 model: %s\n", err.c_str());
                return 1;
            }
            need += eagle3_need;
        }
        uint64_t vulkan_extra = 0;
        if (args.backend == "vulkan") {
            if (!args.dflash_model.empty()) {
                uint64_t ignored_lm_head = 0;
                if (!f16_lm_head_bytes(args.model, &ignored_lm_head, &err)) {
                    std::fprintf(stderr, "error: %s\n", err.c_str());
                    return 1;
                }
                if (dflash_need > std::numeric_limits<uint64_t>::max() - target_need) {
                    std::fprintf(stderr, "error: Vulkan DFlash memory estimate overflow\n");
                    return 1;
                }
                // 安全上界：GPU 常驻完整 target + 完整 DFlash。实际 lm_head 只
                // 上传一次，target/draft 的文件元数据也不会进入 GPU，因此略保守。
                vulkan_extra = dflash_need + target_need;
            } else {
                // 通用 VulkanBackend 会在首次访问时把 FP16 matrix weights 逐个
                // 缓存到 GPU。用完整 target resident 大小作安全上界。
                uint64_t ignored_lm_head = 0;
                if (!f16_lm_head_bytes(args.model, &ignored_lm_head, &err)) {
                    std::fprintf(stderr, "error: %s\n", err.c_str());
                    return 1;
                }
                vulkan_extra = target_need;
            }
            if (need > std::numeric_limits<uint64_t>::max() - vulkan_extra) {
                std::fprintf(stderr, "error: Vulkan memory estimate overflow\n");
                return 1;
            }
            need += vulkan_extra;
        }
        const uint64_t avail = available_memory_bytes();
        const uint64_t budget = avail ? avail - avail / 10 : 0;  // 可用 × 90%
        if (budget && need > budget) {
            std::fprintf(stderr,
                         "error: 模型常驻内存需求 %.0f MB 超过可用上限 %.0f MB\n"
                         "       （当前可用 %.0f MB，留 10%% 余量）。加载会触发换页，\n"
                         "       而 swap 是写操作、消耗 SSD 寿命，故拒绝加载。\n"
                         "       可选修复：\n"
                         "         · 关掉其他大内存进程后重试\n"
                         "         · MoE 模型加 --moe-ssd（专家留盘，resident 大幅缩小）\n"
                         "         · 换更小的量化版本（f16 → i4，常驻约减半）\n"
                         "         · MoE 用 --moe-expert-cache-mb 缩小专家缓存预算\n"
                         "       加 --force-over-memory-budget 可强制继续（不推荐）。\n",
                         need / 1048576.0, budget / 1048576.0, avail / 1048576.0);
            if (!args.force_over_mem_budget) return 1;
            std::fprintf(stderr, "warn: 已按 --force-over-memory-budget 强制继续，"
                                 "可能换页，计时与 SSD 寿命均需自行评估。\n");
        } else {
            std::fprintf(stderr, "[init] mem preflight: need %.0f MB / available %.0f MB "
                         "(上限 %.0f MB) —— 通过\n",
                         need / 1048576.0, avail / 1048576.0, budget / 1048576.0);
        }
        if (vulkan_extra) {
            std::fprintf(stderr,
                         "[init] Vulkan unified-memory copy budget: %.0f MB (included above)\n",
                         vulkan_extra / 1048576.0);
        }
    }

    // ---- 加载权重文件并校验 ----
    // --moe-ssd 时走稀疏加载：路由专家权重一字节都不读进 RAM，只记 offset
    // 交给 ExpertStore 按需 pread。这是 SSD 卸载真正省内存的前提。
    tinyqwen::ModelFile file;
    std::string err;
    if (!file.load(args.model, &err, args.moe_ssd)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    tinyqwen::ModelFile draft_file;
    if (!args.draft_model.empty()) {
        if (!draft_file.load(args.draft_model, &err, false)) {
            std::fprintf(stderr, "error: draft model: %s\n", err.c_str());
            return 1;
        }
        if (draft_file.config().is_moe()) {
            std::fprintf(stderr,
                         "error: MoE draft model is not supported; use a dense draft model\n");
            return 2;
        }
        if (draft_file.config().vocab_size != file.config().vocab_size) {
            std::fprintf(stderr,
                         "error: target/draft vocab mismatch (%u vs %u); they must use "
                         "the same tokenizer\n",
                         file.config().vocab_size, draft_file.config().vocab_size);
            return 2;
        }
        const uint32_t teos = file.config().eos_token_id;
        const uint32_t deos = draft_file.config().eos_token_id;
        if (teos != 0 && deos != 0 && teos != deos) {
            std::fprintf(stderr,
                         "error: target/draft eos mismatch (%u vs %u); tokenizer metadata differs\n",
                         teos, deos);
            return 2;
        }
        std::fprintf(stderr, "[init] speculative draft: %s (K=%d)\n",
                     args.draft_model.c_str(), args.speculative_tokens);
    }
    tinyqwen::ModelFile dflash_file;
    if (!args.dflash_model.empty()) {
        if (!dflash_file.load(args.dflash_model, &err, false)) {
            std::fprintf(stderr, "error: DFlash model: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "[init] DFlash draft: %s (block=%d)\n",
                     args.dflash_model.c_str(), args.speculative_tokens);
    }
    tinyqwen::ModelFile eagle3_file;
    if (!args.eagle3_model.empty()) {
        if (!eagle3_file.load(args.eagle3_model, &err, false)) {
            std::fprintf(stderr, "error: EAGLE3 model: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "[init] EAGLE3 draft: %s (verify width=%d)\n",
                     args.eagle3_model.c_str(), args.speculative_tokens);
    }
    if (args.verbose) file.print_summary();
    if (file.offloaded_count() > 0) {
        const double mb = 1024.0 * 1024.0;
        std::fprintf(stderr, "[init] weights: resident %.2f MB, offloaded %zu tensors / %.2f MB "
                             "on disk (file %.2f MB)\n",
                     file.resident_bytes() / mb, file.offloaded_count(),
                     file.offloaded_bytes() / mb, file.file_bytes() / mb);
    }

    // ---- 解析停止符 eos：CLI 显式值 > 模型文件头（v2）> Qwen2.5 默认 ----
    if (args.eos == -2) {
        const uint32_t hdr_eos = file.config().eos_token_id;
        args.eos = hdr_eos != 0 ? static_cast<int>(hdr_eos) : 151645;
        std::fprintf(stderr, "[init] eos: %d (auto)\n", args.eos);
    }

    // ---- 按模型 dtype 选择 matvec 实现（f32/f16/i4 各有独立注册表）----
    const bool is_f16 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kF16);
    const bool is_i4 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kI4);
    const bool is_vq2 = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kVQ2);
    const bool is_gptq = file.header().dtype == static_cast<uint32_t>(tinyqwen::Dtype::kGPTQ4);
    if (is_vq2) {
        if (!tinyqwen::set_matvec_vq2_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr,
                         "error: unknown vq2 matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_vq2_impls());
            return 2;
        }
        if (!tinyqwen::set_matvec_impl_by_name(impl_name.c_str())) {
            tinyqwen::set_matvec_impl_by_name("neon_mt_kv_nt");
        }
        // embed/lm_head 存 f16 时，f16 注册表也要选优化内核（否则 lm_head 落 ref）
        if (!tinyqwen::set_matvec_f16_impl_by_name(impl_name.c_str())) {
            tinyqwen::set_matvec_f16_impl_by_name("neon_mt_kv_nt");
        }
        // 同理，embed/lm_head 存紧凑 INT4 时 i4 注册表也要选优化内核
        if (!tinyqwen::set_matvec_i4_impl_by_name(impl_name.c_str())) {
            tinyqwen::set_matvec_i4_impl_by_name("sdot4_mt");
        }
        std::fprintf(stderr,
                     "[init] matvec impl: %s (vq2 weights; lm_head f32: %s / f16: %s / i4: %s)\n",
                     tinyqwen::matvec_vq2_impl_name(), tinyqwen::matvec_impl_name(),
                     tinyqwen::matvec_f16_impl_name(), tinyqwen::matvec_i4_impl_name());
    } else if (is_i4) {
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
        if (!tinyqwen::set_matmul_f16_impl_by_name(impl_name.c_str()))
            tinyqwen::set_matmul_f16_impl_by_name("ref");
        std::fprintf(stderr, "[init] matvec impl: %s; matmul: %s (f16 weights)\n",
                     tinyqwen::matvec_f16_impl_name(), tinyqwen::matmul_f16_impl_name());
    } else if (is_gptq) {
        // GPTQ 有独立注册表（AutoGPTQ 列主序，与 i4 的 HQQ interleaved 不兼容）。
        // 没有这个分支时 GPTQ 模型会掉进下面的 else，选到 f32 注册表——
        // 名字能对上但语义完全错，且 CLI 报的可用列表里永远看不到 GPTQ 变体。
        if (!tinyqwen::set_matvec_gptq_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr,
                         "error: unknown gptq matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_gptq_impls());
            return 2;
        }
        // router 是 fp32、lm_head/embed 保留源 dtype（真 checkpoint 里是 fp16），
        // 所以三个注册表都要设优化内核。**不传播用户的 impl 名**：GPTQ 的
        // "neon"/"neon_mt"/"ref" 与 f32/f16 注册表同名但语义完全不同——f32 的
        // "neon" 是单线程版。传播会让 lm_head（每 token 全读一遍）落到单线程内核。
        // 用户选的是 GPTQ kernel，其余 dtype 一律用各自族的最佳实现。
        // 漏设 f16 表会让 fp16 lm_head 落到标量 ref（坑 #21 同类陷阱）。
        if (!tinyqwen::set_matvec_impl_by_name("neon_mt_kv_nt")) {
            tinyqwen::set_matvec_impl_by_name("neon_mt");
        }
        if (!tinyqwen::set_matvec_f16_impl_by_name("neon_mt_kv_nt")) {
            tinyqwen::set_matvec_f16_impl_by_name("ref");
        }
        if (!tinyqwen::set_matmul_f16_impl_by_name("neon_mt_kv_nt"))
            tinyqwen::set_matmul_f16_impl_by_name("ref");
        // 批量 GEMM（MoE 批量 prefill）用 NEON 版；标量 ref 会让 prefill
        // 反而比逐 token 慢 3.5×（实测 expert_ffn 25.14s vs 4.50s）。
        tinyqwen::set_matmul_gptq_impl_by_name("neon");
        std::fprintf(stderr,
                     "[init] matvec impl: %s (gptq4 weights, group=%u; router/lm_head f32: %s)\n",
                     tinyqwen::matvec_gptq_impl_name(), file.config().quant_group_size,
                     tinyqwen::matvec_impl_name());
    } else {
        if (!tinyqwen::set_matvec_impl_by_name(impl_name.c_str())) {
            std::fprintf(stderr, "error: unknown matvec_impl '%s' (available: %s)\n",
                         impl_name.c_str(), tinyqwen::available_matvec_impls());
            return 2;
        }
        std::fprintf(stderr, "[init] matvec impl: %s\n", tinyqwen::matvec_impl_name());
    }

    // 各 dtype 有独立注册表。目标 f16 + 草稿 i4 时可分别选 kernel；若两者
    // dtype 相同，这个选项会覆盖该 dtype 的全局实现（两边共同使用）。
    if (!args.draft_model.empty() && !args.draft_matvec_impl.empty()) {
        const char *name = args.draft_matvec_impl.c_str();
        const auto dt = static_cast<tinyqwen::Dtype>(draft_file.header().dtype);
        bool ok = false;
        const char *available = "";
        if (dt == tinyqwen::Dtype::kF16) {
            ok = tinyqwen::set_matvec_f16_impl_by_name(name);
            available = tinyqwen::available_matvec_f16_impls();
        } else if (dt == tinyqwen::Dtype::kI4) {
            ok = tinyqwen::set_matvec_i4_impl_by_name(name);
            available = tinyqwen::available_matvec_i4_impls();
        } else if (dt == tinyqwen::Dtype::kVQ2) {
            ok = tinyqwen::set_matvec_vq2_impl_by_name(name);
            available = tinyqwen::available_matvec_vq2_impls();
        } else if (dt == tinyqwen::Dtype::kGPTQ4) {
            ok = tinyqwen::set_matvec_gptq_impl_by_name(name);
            available = tinyqwen::available_matvec_gptq_impls();
        } else {
            ok = tinyqwen::set_matvec_impl_by_name(name);
            available = tinyqwen::available_matvec_impls();
        }
        if (!ok) {
            std::fprintf(stderr,
                         "error: unknown draft matvec impl '%s' (available: %s)\n",
                         name, available);
            return 2;
        }
        std::fprintf(stderr, "[init] draft matvec impl: %s\n", name);
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
    tinyqwen::Profiler draft_profiler(false);
    const bool is_qwen35 = file.config().uses_qwen35_attention();
    profiler.set_meta(is_qwen35 ? "qwen3.5-hybrid" : "qwen2.5-like",
                      args.backend == "vulkan" ? "vulkan_fp16_hybrid" : "cpu_ref",
                      is_f16 ? "f16w_fp32a" : "fp32");

    // ---- 创建后端（默认 CPU，可选 CUDA / Android Vulkan）----
    std::unique_ptr<tinyqwen::IBackend> backend;
    const bool use_vulkan_backend = args.backend == "vulkan";
    const bool use_vulkan_dflash = use_vulkan_backend && !args.dflash_model.empty();
    if (args.backend == "cuda") {
#ifdef TINYQWEN_HAS_CUDA
        backend = tinyqwen::create_cuda_backend();
        std::fprintf(stderr, "[init] backend: CUDA\n");
#else
        std::fprintf(stderr, "error: --backend cuda requires CUDA build\n");
        return 1;
#endif
    } else if (use_vulkan_backend) {
        if (use_vulkan_dflash) {
            // Prompt prefill 仍由 CPU 完成并作为精确基线；随后把 KV 前缀同步
            // 一次，DFlash proposal 与 target verify 都在同一 Vulkan device。
            backend = tinyqwen::create_cpu_backend();
            std::fprintf(stderr,
                         "[init] backend: CPU prefill + Vulkan DFlash/target decode\n");
        } else {
            std::string verr;
            backend = tinyqwen::create_vulkan_backend(&verr);
            if (!backend) {
                std::fprintf(stderr, "error: Vulkan backend create failed: %s\n", verr.c_str());
                return 1;
            }
            std::fprintf(stderr,
                         "[init] backend: Vulkan FP16 matrices + CPU stateful ops\n");
        }
    } else if (args.backend.empty()) {
        backend = tinyqwen::create_cpu_backend();
    } else {
        std::fprintf(stderr,
                     "error: unknown backend '%s' (available: cpu, cuda, vulkan)\n",
                     args.backend.c_str());
        return 2;
    }

    // ---- 建模：校验权重、分配 KV cache 和 workspace ----
    // MoE：ExpertStore 在 create() 之前打开（create 注册专家 offset 进去），
    // 运行时由 --moe-ssd 决定走 pread 还是 resident 指针。
    tinyqwen::ExpertStore expert_store;
    if (file.config().is_moe()) {
        if (!expert_store.open(args.model, &err)) {
            std::fprintf(stderr, "error: %s\n", err.c_str());
            return 1;
        }
        expert_store.set_cache_slots(args.moe_cache_slots);
    }
    std::unique_ptr<tinyqwen::QwenModel> model;
    if (!tinyqwen::QwenModel::create(file, args.max_seq_len, profiler, &err, &model,
                                     std::move(backend), args.kv_fp16,
                                     file.config().is_moe() ? &expert_store : nullptr)) {
        std::fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    std::unique_ptr<tinyqwen::QwenModel> draft_model;
    if (!args.draft_model.empty()) {
        if (!tinyqwen::QwenModel::create(draft_file, args.max_seq_len, draft_profiler,
                                         &err, &draft_model,
                                         tinyqwen::create_cpu_backend(), args.kv_fp16)) {
            std::fprintf(stderr, "error: draft model: %s\n", err.c_str());
            return 1;
        }
    }
    std::unique_ptr<tinyqwen::DFlashModel> dflash_model;
    if (!args.dflash_model.empty()) {
        if (!tinyqwen::DFlashModel::create(dflash_file, args.max_seq_len, *model,
                                           &err, &dflash_model)) {
            std::fprintf(stderr, "error: DFlash model: %s\n", err.c_str());
            return 1;
        }
        if (args.speculative_tokens > dflash_model->block_size()) {
            std::fprintf(stderr,
                         "error: --speculative-tokens %d exceeds DFlash block size %d\n",
                         args.speculative_tokens, dflash_model->block_size());
            return 2;
        }
        if (use_vulkan_dflash && !dflash_model->enable_vulkan(&err)) {
            std::fprintf(stderr, "error: DFlash Vulkan engine: %s\n", err.c_str());
            return 1;
        }
    }
    std::unique_ptr<tinyqwen::Eagle3Model> eagle3_model;
    if (!args.eagle3_model.empty()) {
        if (!tinyqwen::Eagle3Model::create(eagle3_file, args.max_seq_len, *model,
                                           &err, &eagle3_model,
                                           tinyqwen::create_cpu_backend())) {
            std::fprintf(stderr, "error: EAGLE3 model: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr,
                     "[init] EAGLE3: draft_vocab=%d target_layers=%d,%d,%d\n",
                     eagle3_model->draft_vocab_size(),
                     eagle3_model->target_layer_ids()[0],
                     eagle3_model->target_layer_ids()[1],
                     eagle3_model->target_layer_ids()[2]);
    }
    if (file.config().is_moe()) {
        model->set_moe_ssd(args.moe_ssd);

        // 字节预算必须在 create() 之后设：create() 才把专家 offset/nbytes 注册进
        // store，per_expert_bytes_ 在那之前是 0，无法把"预算 MB"换算成槽数。
        // 上面先按 slots 设一次默认值，这里若指定了预算则覆盖。
        int effective_slots = args.moe_cache_slots;
        if (args.moe_cache_mb > 0) {
            const uint64_t budget =
                static_cast<uint64_t>(args.moe_cache_mb) * 1024ull * 1024ull;
            if (!expert_store.set_cache_budget(budget, &err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 1;
            }
            effective_slots = static_cast<int>(budget / expert_store.per_expert_bytes());
        }

        // B-2 异步预取：必须在 create() 之后启用（需要单专家字节数分配缓冲）。
        // 槽数须 ≥ experts_per_tok，否则本层内槽复用会让先预取的被覆盖。
        if (args.moe_ssd && args.moe_prefetch) {
            const int want = args.moe_prefetch_slots;
            const int need = model->config().num_experts_per_tok;
            if (want < need) {
                std::fprintf(stderr,
                             "error: --moe-prefetch-slots %d < experts_per_tok %d —— "
                             "本层内槽会复用、先预取的被覆盖。请设 ≥ %d。\n",
                             want, need, need);
                return 1;
            }
            if (!expert_store.prefetch_enable(want, &err)) {
                std::fprintf(stderr, "error: %s\n", err.c_str());
                return 1;
            }
        }

        // 专家级并行必须在 create() 之后设：staging/workspace 按 experts_per_tok_
        // 与 per_expert_bytes 分配。与预取互斥——并行路径绕开 LRU 直读到 staging，
        // 预取槽不会被消费，同时开等于白分配一块预取池。
        if (args.moe_expert_threads > 0) {
            if (args.moe_prefetch) {
                std::fprintf(stderr,
                             "error: --moe-expert-threads 与 --moe-prefetch 互斥 —— "
                             "并行路径绕开 LRU 直读 staging，预取槽不会被消费。\n");
                return 1;
            }
            model->set_moe_expert_threads(args.moe_expert_threads);
        }

        // 内存上限校验：resident + 专家缓存预算 + 预取缓冲池 不得超过**当前可用**
        // 内存。为什么不用物理总量：wired + 其他进程已占掉大半，实测 16 GB 机器
        // wired 就有 8 GB、可用只剩 1.5 GB。按物理总量校验会宽松 7×，放行后照样
        // 换页。swap 是**写**操作、消耗 SSD 寿命，所以宁可 fail-fast 也不要静默换页。
        // 留 10% 余量：进程自身还有 KV cache / workspace / 栈等未计入的开销。
        // 并行模式不调 get()，LRU 槽缓冲（懒分配）永不增长，故不计入预算；
        // 若照常计入会凭空多出 slots × per_expert_bytes 的幻影占用，
        // 大预算下足以误触发 fail-fast。
        const uint64_t cache_bytes = model->moe_expert_threads() > 0
                                         ? 0
                                         : static_cast<uint64_t>(effective_slots) *
                                               expert_store.per_expert_bytes();
        const uint64_t pf_bytes = expert_store.prefetch_pool_bytes();
        const uint64_t ep_bytes = model->moe_expert_parallel_bytes();
        const uint64_t total_bytes =
            file.resident_bytes() + cache_bytes + pf_bytes + ep_bytes;
        const uint64_t phys = physical_memory_bytes();
        const uint64_t avail = available_memory_bytes();
        const uint64_t budget = avail ? avail - avail / 10 : 0;  // 可用 × 90%
        std::fprintf(stderr,
                     "[init] moe: experts=%d per_tok=%d ssd=%s cache_slots=%d "
                     "prefetch=%s expert_threads=%d\n"
                     "[init] mem budget: resident %.0f MB + expert cache %.0f MB "
                     "+ prefetch pool %.0f MB + parallel staging %.0f MB = %.0f MB\n"
                     "[init] mem available: %.0f MB (物理 %.0f MB), 留 10%% 余量后 "
                     "上限 %.0f MB\n",
                     model->config().n_routed_experts, model->config().num_experts_per_tok,
                     args.moe_ssd ? "on" : "off", effective_slots,
                     args.moe_prefetch ? "on" : "off", model->moe_expert_threads(),
                     file.resident_bytes() / 1048576.0, cache_bytes / 1048576.0,
                     pf_bytes / 1048576.0, ep_bytes / 1048576.0, total_bytes / 1048576.0,
                     avail / 1048576.0, phys / 1048576.0, budget / 1048576.0);
        if (budget && total_bytes > budget) {
            std::fprintf(stderr,
                         "error: 内存需求 %.0f MB 超过可用上限 %.0f MB（可用 %.0f MB "
                         "留 10%% 余量）—— 会触发换页，而 swap 是写操作、消耗 SSD 寿命。\n"
                         "       请先释放内存（关掉其他大进程），或用 "
                         "--moe-expert-cache-mb 缩小专家缓存预算。\n"
                         "       加 --force-over-memory-budget 可强制继续（不推荐）。\n",
                         total_bytes / 1048576.0, budget / 1048576.0, avail / 1048576.0);
            if (!args.force_over_mem_budget) return 1;
            std::fprintf(stderr, "warn: 已按 --force-over-memory-budget 强制继续，"
                                 "可能换页，计时与 SSD 寿命均需自行评估。\n");
        }
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

    // ---- 选择 engine（GPU 上的整段 forward，opt-in）----
    //
    // 两个引擎管不同阶段，互斥：cuda 管 decode，metal 管 prefill。
    // metal 接管 prefill 后 decode 仍走 CPU：它把 post-RoPE 的 K/V 写进
    // model->kv_cache() 并 advance(n)，CPU decode 才能从位置 n 接续。
    std::string engine_name = args.engine;
    if (engine_name.empty()) engine_name = config.get("engine", "");
    tinyqwen::GpuDecodeEngine *engine = nullptr;
    tinyqwen::MetalPrefillEngine *metal_engine = nullptr;
    if (!engine_name.empty()) {
        std::string eerr;
        if (engine_name == "metal") {
            if (!tinyqwen::metal_prefill_available()) {
                std::fprintf(stderr, "error: --engine metal 仅在 Apple 平台可用\n");
                return 2;
            }
            if (args.topk > 0) {
                std::fprintf(stderr, "error: --topk 在 --engine metal 下暂不支持（用 --dump-logits）\n");
                return 2;
            }
            if (!tinyqwen::metal_prefill_create(&file, args.max_seq_len, &eerr, &metal_engine)) {
                std::fprintf(stderr, "error: metal engine create failed: %s\n", eerr.c_str());
                return 1;
            }
            std::fprintf(stderr, "[init] prefill engine: metal (%s)\n",
                         tinyqwen::metal_prefill_device_name(metal_engine).c_str());
        } else {
            if (!tinyqwen::set_gpu_decode_impl_by_name(engine_name.c_str())) {
                std::fprintf(stderr, "error: unknown engine '%s' (available: cuda, metal)\n",
                             engine_name.c_str());
                return 2;
            }
            if (args.topk > 0 || !args.dump_logits.empty()) {
                std::fprintf(stderr,
                             "error: --topk / --dump-logits 在 --engine 下暂不支持（v1）\n");
                return 2;
            }
            if (!tinyqwen::gpu_decode_create(&file, args.max_seq_len, &eerr, &engine)) {
                std::fprintf(stderr, "error: gpu engine create failed: %s\n", eerr.c_str());
                return 1;
            }
            std::fprintf(stderr, "[init] decode engine: %s (GPU-resident forward)\n",
                         tinyqwen::gpu_decode_impl_name());
        }
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
    // --ppl --ppl-jsonl 模式下序列来自 ppl_jsonl，此处无需解析
    std::vector<int> tokens;
    const bool ppl_jsonl_only = args.ppl && !args.ppl_jsonl.empty();
    if (!ppl_jsonl_only) {
        tokens = args.tokens_csv.empty() ? parse_tokens_json(args.tokens_json)
                                         : parse_csv(args.tokens_csv);
        if (tokens.empty()) {
            std::fprintf(stderr, "error: empty token list\n");
            return 2;
        }
    }
    // 融合开关：CLI > 配置文件 > 默认 true
    const bool fuse_gate_up = args.no_fuse_gate_up ? false
                              : config.get("fuse_gate_up", "true") != "false";
    const bool fuse_qkv = args.no_fuse_qkv ? false
                           : config.get("fuse_qkv", "true") != "false";
    model->set_fuse_gate_up(fuse_gate_up);
    model->set_fuse_qkv(fuse_qkv);
    if (draft_model) {
        draft_model->set_fuse_gate_up(fuse_gate_up);
        draft_model->set_fuse_qkv(fuse_qkv);
    }
    // Qwen3.5 批量 prefill 开关：CLI > 配置文件 > 默认 true
    const bool batch_prefill = args.no_batch_prefill ? false
                               : config.get("batch_prefill", "true") != "false";
    model->set_batch_prefill(batch_prefill);
    model->set_prompt_len(static_cast<int>(tokens.size()));
    if (draft_model) {
        draft_model->set_batch_prefill(batch_prefill);
        draft_model->set_prompt_len(static_cast<int>(tokens.size()));
    }

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
    const auto dump = [&](const float *lg) {
        if (logits_out) {
            std::fwrite(lg, sizeof(float), file.config().vocab_size, logits_out);
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

    // ---- PPL 模式：teacher-forcing 困惑度（不生成）----
    if (args.ppl) {
        std::vector<std::vector<int>> seqs;
        if (!args.ppl_jsonl.empty()) {
            seqs = parse_batch_jsonl(args.ppl_jsonl);
        } else {
            seqs.push_back(tokens); // 单序列：来自 --tokens / --tokens-json
        }
        double weighted_nll = 0.0; // Σ (nll_i × count_i)
        long total_count = 0;
        for (size_t si = 0; si < seqs.size(); ++si) {
            const std::vector<int> &tok = seqs[si];
            const int n = static_cast<int>(tok.size());
            if (n < 2) {
                std::fprintf(stderr, "[ppl] seq %zu 太短(%d)，跳过\n", si, n);
                continue;
            }
            if (n > args.max_seq_len) {
                std::fprintf(stderr, "error: seq %zu 长度 %d > max_seq_len %d（加大 --max-seq-len）\n",
                             si, n, args.max_seq_len);
                return 2;
            }
            model->reset();
            model->set_prompt_len(n);
            long cnt = 0;
            const double nll = model->forward_ppl(tok.data(), n, &cnt);
            weighted_nll += nll * static_cast<double>(cnt);
            total_count += cnt;
        }
        if (total_count == 0) {
            std::fprintf(stderr, "error: 没有可计分的 token\n");
            return 1;
        }
        const double mean_nll = weighted_nll / static_cast<double>(total_count);
        std::printf("ppl_nll %.6f\n", mean_nll);
        std::printf("ppl %.6f\n", std::exp(mean_nll));
        std::printf("ppl_tokens %ld\n", total_count);
        if (logits_out) std::fclose(logits_out);
        return 0;
    }

    // ---- 精确 greedy 投机解码 ----
    if (draft_model || dflash_model || eagle3_model) {
        tinyqwen::SpeculativeConfig spec_cfg;
        spec_cfg.max_new_tokens = args.max_new_tokens;
        spec_cfg.draft_tokens = args.speculative_tokens;
        spec_cfg.eos_token_id = args.eos;
        spec_cfg.batch_target_verify = !args.no_eagle3_batch_verify;
        tinyqwen::SpeculativeResult spec;
        std::string serr;
        bool ok = false;
        if (dflash_model) {
            ok = tinyqwen::dflash_speculative_generate(*model, *dflash_model, tokens,
                                                        spec_cfg, &spec, &serr);
        } else if (eagle3_model) {
            ok = tinyqwen::eagle3_speculative_generate(*model, *eagle3_model, tokens,
                                                        spec_cfg, &spec, &serr);
        } else {
            ok = tinyqwen::speculative_generate(*model, *draft_model, metal_engine,
                                                 tokens, spec_cfg, &spec, &serr);
        }
        if (!ok) {
            std::fprintf(stderr, "error: speculative decoding failed: %s\n", serr.c_str());
            if (metal_engine) tinyqwen::metal_prefill_destroy(metal_engine);
            return 1;
        }

        for (size_t i = 0; i < spec.generated_ids.size(); ++i)
            std::printf("gen %zu %d\n", i, spec.generated_ids[i]);
        std::printf("generated_ids:");
        for (int id : spec.generated_ids) std::printf(" %d", id);
        std::printf("\n");

        const tinyqwen::SpeculativeStats &s = spec.stats;
        std::fprintf(stderr,
                     "[speculative] generated=%zu blocks=%d proposed=%d accepted=%d "
                     "acceptance=%.1f%% corrections=%d bonus=%d rollbacks=%d\n"
                     "[speculative] target_calls=%d target_inputs=%d draft_calls=%d "
                     "prefill=%.2f ms decode=%.2f ms draft=%.2f ms verify=%.2f ms\n",
                     spec.generated_ids.size(), s.blocks, s.draft_proposed,
                     s.draft_accepted, s.acceptance_rate() * 100.0,
                     s.corrections, s.bonus_tokens, s.rollbacks,
                     s.target_verify_calls, s.target_input_tokens,
                     s.draft_forward_calls, s.prefill_ms, s.decode_ms,
                     s.draft_ms, s.target_verify_ms);

        if (!args.speculative_stats_out.empty()) {
            FILE *sf = std::fopen(args.speculative_stats_out.c_str(), "w");
            if (!sf) {
                std::fprintf(stderr, "error: cannot write %s\n",
                             args.speculative_stats_out.c_str());
                if (metal_engine) tinyqwen::metal_prefill_destroy(metal_engine);
                return 1;
            }
            std::fprintf(sf,
                         "{\n"
                         "  \"generated_tokens\": %zu,\n"
                         "  \"hit_eos\": %s,\n"
                         "  \"draft_tokens_per_block\": %d,\n"
                         "  \"target_verify_mode\": \"%s\",\n"
                         "  \"blocks\": %d,\n"
                         "  \"draft_proposed\": %d,\n"
                         "  \"draft_accepted\": %d,\n"
                         "  \"acceptance_rate\": %.8f,\n"
                         "  \"corrections\": %d,\n"
                         "  \"bonus_tokens\": %d,\n"
                         "  \"rollbacks\": %d,\n"
                         "  \"target_verify_calls\": %d,\n"
                         "  \"target_input_tokens\": %d,\n"
                         "  \"draft_forward_calls\": %d,\n"
                         "  \"baseline_tail_steps\": %d,\n"
                         "  \"prefill_ms\": %.6f,\n"
                         "  \"draft_ms\": %.6f,\n"
                         "  \"target_verify_ms\": %.6f,\n"
                         "  \"decode_ms\": %.6f\n"
                         "}\n",
                         spec.generated_ids.size(), spec.hit_eos ? "true" : "false",
                         args.speculative_tokens,
                         args.no_eagle3_batch_verify ? "sequential" : "batched",
                         s.blocks, s.draft_proposed,
                         s.draft_accepted, s.acceptance_rate(), s.corrections,
                         s.bonus_tokens, s.rollbacks, s.target_verify_calls,
                         s.target_input_tokens, s.draft_forward_calls,
                         s.baseline_tail_steps, s.prefill_ms, s.draft_ms,
                         s.target_verify_ms, s.decode_ms);
            std::fclose(sf);
            std::fprintf(stderr, "[speculative-stats] %s\n",
                         args.speculative_stats_out.c_str());
        }
        if (metal_engine) tinyqwen::metal_prefill_destroy(metal_engine);
        return 0;
    }

    // ---- prefill 阶段 ----
    using InferenceClock = std::chrono::steady_clock;
    const auto prefill_wall_begin = InferenceClock::now();
    int next = 0;
    tinyqwen::TopKResult topk;
    const size_t vocab = file.config().vocab_size;
    std::vector<float> metal_logits;
    if (engine) {
        // GPU-resident engine 没有批量 prefill 入口，只能逐 token 喂进去
        for (size_t i = 0; i < tokens.size(); ++i) {
            next = engine_step(tokens[i], true);
            if (args.verbose) {
                std::fprintf(stderr, "[prefill] %zu/%zu id=%d -> next=%d\n", i + 1, tokens.size(),
                             tokens[i], next);
            }
        }
    } else if (metal_engine) {
        // Apple GPU 批量 prefill：整批一次前向。post-RoPE 的 K/V 写进 kv_cache
        // 并 advance(n)，后面的 decode 仍走 CPU 路径。
        // dump 口径不变：默认只写末位一行（与 CPU 批量 prefill 一致）；--verbose
        // 写全部 n 行，供逐位置对齐用。
        // 区别在于现在按这个口径**按需计算**：只有"verbose 且真的要 dump"才让引擎
        // 算全部 n 行，否则只算末位一行 —— 省掉 lm_head 的全行投影
        // （seq=512 时 65.6 ms → ~3 ms），见 docs/optimization_log.md。
        const bool need_all_rows = args.verbose && logits_out != nullptr;
        if (logits_out) metal_logits.assign(need_all_rows ? tokens.size() * vocab : vocab, 0.0f);
        float *lm_buf = logits_out ? metal_logits.data() : nullptr;
        std::string merr;
        profiler.begin_token(0, 0, /*is_prefill=*/true);
        next = tinyqwen::metal_prefill_run(metal_engine, tokens.data(),
                                           static_cast<int>(tokens.size()), lm_buf, need_all_rows,
                                           &model->kv_cache(), &model->gdn_state(), &merr);
        profiler.end_token();
        if (next < 0) {
            std::fprintf(stderr, "error: metal prefill failed: %s\n", merr.c_str());
            if (logits_out) std::fclose(logits_out);
            tinyqwen::metal_prefill_destroy(metal_engine);
            return 1;
        }
        if (logits_out) {
            // 末行路径下引擎把末位 logits 写在 lm_buf[0..vocab)
            if (need_all_rows) {
                for (size_t i = 0; i < tokens.size(); ++i)
                    dump(metal_logits.data() + i * vocab);
            } else {
                dump(metal_logits.data());
            }
        }
    } else if (args.verbose) {
        // 逐 token prefill：可打印每步详情
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool last_prefill = i + 1 == tokens.size();
            const bool need_topk = args.topk > 0 && last_prefill;
            next = model->forward_token(tokens[i], need_topk ? &topk : nullptr, args.topk);
            dump(model->last_logits());
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
        // --dump-logits 时补一份 prefill 末位 logits（批量 prefill 对齐验证用；
        // 此前非 verbose 路径只 dump decode 位。现有对齐脚本走 --verbose，不受影响）
        dump(model->last_logits());
    }
    std::fprintf(stderr, "[prefill] %zu tokens done\n", tokens.size());
    const auto prefill_wall_end = InferenceClock::now();
    if (args.topk > 0) print_topk(topk);

    // ---- decode 阶段 ----
    const auto decode_wall_begin = InferenceClock::now();
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
            dump(model->last_logits());
        }
        if (args.topk > 0) print_topk(topk);
    }

    std::printf("generated_ids:");
    for (int id: generated) std::printf(" %d", id);
    std::printf("\n");
    const auto decode_wall_end = InferenceClock::now();
    const double prefill_wall_ms =
        std::chrono::duration<double, std::milli>(prefill_wall_end - prefill_wall_begin).count();
    const double decode_wall_ms =
        std::chrono::duration<double, std::milli>(decode_wall_end - decode_wall_begin).count();
    std::fprintf(stderr, "[timing] prefill=%.2f ms decode=%.2f ms\n",
                 prefill_wall_ms, decode_wall_ms);

    profiler.set_counts(tokens.size(), generated.size());

    if (engine) tinyqwen::gpu_decode_destroy(engine);
    if (metal_engine) tinyqwen::metal_prefill_destroy(metal_engine);

    if (!args.profile_out.empty()) {
        std::string werr;
        if (!profiler.write_json(args.profile_out, &werr)) {
            std::fprintf(stderr, "error: %s\n", werr.c_str());
            return 1;
        }
        std::fprintf(stderr, "[profile] %s\n", args.profile_out.c_str());
    }
    if (logits_out) std::fclose(logits_out);
    // MoE SSD 卸载统计（命中/miss/淘汰/读字节）——机制归因用
    if (file.config().is_moe() && args.moe_ssd) {
        const auto s = model->moe_stats();
        std::fprintf(stderr,
                     "[moe] expert cache: hits=%zu misses=%zu evictions=%zu bytes_read=%zu\n",
                     s.hits, s.misses, s.evictions, s.bytes_read);
        if (s.pf_hits + s.pf_waits + s.pf_fallbacks > 0) {
            const size_t tot = s.pf_hits + s.pf_waits + s.pf_fallbacks;
            std::fprintf(stderr,
                         "[moe] prefetch: hits=%zu (%.0f%%) waits=%zu (%.0f%%) "
                         "fallbacks=%zu (%.0f%%) wait_ms=%.0f\n",
                         s.pf_hits, s.pf_hits * 100.0 / tot,
                         s.pf_waits, s.pf_waits * 100.0 / tot,
                         s.pf_fallbacks, s.pf_fallbacks * 100.0 / tot,
                         s.pf_wait_ms);
        }
    }
    return 0;
}
