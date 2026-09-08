// ============================================================================
// 文件: expert_store.h
// 作用: MoE 路由专家的 SSD 卸载存储 —— 按需 pread + LRU 缓存
//
// 核心命题（本项目要验证的）：
//   35B MoE 权重 ~17.5GB 装不进内存；但每个 token 只激活 k 个专家。
//   把路由专家权重留在 .tqwen 文件（SSD），只有被路由门选中的专家才
//   pread 进一个固定大小的 LRU 缓存参与计算。未激活的专家永远不进 RAM。
//
// 设计要点：
//   - pread 而非 mmap：mmap 的 readahead 会把未激活专家也读进 page cache，
//     违背"只读激活专家"的命题；pread 精确只读所需字节。
//   - LRU 槽缓存：槽数 = 可容纳专家数（每个专家 = gate/up/down 三块）。
//     命中返零拷贝指针；未命中淘汰最久未用槽 + pread 三块。
//   - 共享专家 / 路由门 / attention 等不在本 store（resident，在 ModelFile）。
//
// 与 ModelFile 的关系：
//   --moe-ssd 时 ModelFile 走稀疏加载：data_ 只含 header + tensor 表 + resident
//   tensor，路由专家的字节**一整个不进 RAM**（其 TensorView.data 为 nullptr）。
//   forward 只经 ExpertStore::get() 取专家权重。offset/nbytes 在 create() 时
//   直接取 TensorView::file_offset 注册——稀疏紧凑重排后 (view.data - base())
//   不再等于文件内偏移，指针算式会算错。
//   不开 --moe-ssd 时 ModelFile 仍整文件读入，专家走 resident 指针（正确性锚点）。
//
// 生命周期：返回的 ExpertWeights 指针指向缓存槽内部缓冲，**仅在当前 forward
//   步内有效**（下一 token 可能淘汰该槽）。同步 get() 下不跨 token 持有。
// ============================================================================
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tinyqwen {

    // 一个被激活专家的三块权重视图（指针指向 ExpertStore 缓存槽内部缓冲）
    struct ExpertWeights {
        const uint8_t *gate = nullptr;  // [inter, hidden] GPTQ in-band 块
        const uint8_t *up = nullptr;    // [inter, hidden]
        const uint8_t *down = nullptr;  // [hidden, inter]
    };

    struct ExpertStoreStats {
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t bytes_read = 0;  // 累计 pread 字节（不含命中的零拷贝）
        // B-2 异步预取归因（与 hits/misses 分开记，避免归因失真）
        size_t pf_hits = 0;     // get() 命中已就绪的预取槽（零等待）
        size_t pf_waits = 0;    // get() 等待在飞的预取（部分重叠）
        size_t pf_fallbacks = 0;// 未预取，回退同步 LRU pread
        double pf_wait_ms = 0.0;// 累计等待预取的墙钟时间
    };

    class ExpertStore {
    public:
        ExpertStore() = default;
        ~ExpertStore();

        ExpertStore(const ExpertStore &) = delete;
        ExpertStore &operator=(const ExpertStore &) = delete;

        // 打开 .tqwen 文件供后续 pread（只读）
        bool open(const std::string &path, std::string *err);

        // 注册一个专家的三块权重（offset = 文件内绝对偏移，nbytes = 块字节数）。
        // create() 遍历专家 tensor 调用本函数建表。inter/hidden/group_size 供
        // 调用方（forward）取维度，store 本身只按 nbytes 读字节。
        void register_expert(int layer, int expert,
                             uint64_t gate_off, uint64_t gate_nbytes,
                             uint64_t up_off, uint64_t up_nbytes,
                             uint64_t down_off, uint64_t down_nbytes,
                             int inter, int hidden, int group_size);

        // 设置 LRU 缓存槽数（= 可同时驻留的专家数）。0 = 全 miss（每次都 pread）。
        void set_cache_slots(int n);

        // 按**字节预算**设置缓存容量（而非槽数）。槽数 = budget / 单个专家字节数。
        // 为什么需要它：按槽数限界时，用户填一个大数（如 4096）会让 cache 悄悄
        // 吃掉 ~10 GB，把机器推进重度换页（实测 swap used 11.3 GB / 12 GB），
        // 此后所有计时不可信。字节预算让上限显式、可审计。
        // 单个专家字节数超过预算时 fail-fast（返回 false）——那种配置永远装不下
        // 一个专家，静默降级成 slots=0 会让用户以为预算生效了。
        // 必须在 register_expert() 之后调用（需要知道单专家字节数）。
        bool set_cache_budget(uint64_t budget_bytes, std::string *err);

        // 归因访问器：当前缓存占用的字节上限与推算出的槽数
        uint64_t cache_budget_bytes() const { return cache_budget_bytes_; }
        uint64_t per_expert_bytes() const { return per_expert_bytes_; }
        // load_expert_direct 所需的 staging 字节数（连续布局为跨度，含对齐填充，
        // 比 per_expert_bytes 大）。分配直读缓冲必须用这个，不能用前者。
        uint64_t expert_staging_bytes() const { return expert_staging_bytes_; }

        // 取一个专家的三块权重：命中缓存则零拷贝返回；未命中淘汰 LRU + pread。
        // 返回的指针指向缓存槽内部，仅在下次淘汰前有效。
        const ExpertWeights get(int layer, int expert);

        // 并行直读路径不经 stats_（裸 size_t，并发累加不安全），改用独立原子计数，
        // 读时并入 bytes_read。否则并行模式下 hits/misses/bytes_read 全为 0，
        // 有效带宽归因（docs 里那套 GB/s 反推法）会失效。
        ExpertStoreStats stats() const {
            ExpertStoreStats s = stats_;
            s.bytes_read += direct_bytes_read_.load(std::memory_order_relaxed);
            return s;
        }

        // 从 .tqwen 在 offset 处读 nbytes 字节到 out（调用方分配）。
        // 供 embed_tokens 卸载后按需读单行用：embed 是查表，每 token 只读 1 行
        // （hidden*4 = 8 KB），却占 1187 MB fp32（本模型 resident 的 41%）。
        // 复用本 store 已持有的 fd，不必再开一个。
        bool read_bytes(uint64_t offset, size_t nbytes, void *out);

        // 绕过 LRU，直接 pread 一个专家到调用方缓冲。**线程安全**：::pread 无文件
        // 偏移状态，layout_map_ 在 register_expert 后只读，各调用方写自己的缓冲。
        // 供专家级并行用——get() 返回的指针指向 LRU 槽，会被后续淘汰覆盖，无法同时
        // 持有 top-k 个专家；而实测 LRU 对 MoE decode 的顺序扫描 hits=0（坑 #34），
        // 绕过它不损失任何命中率。contiguous 时合并成 1 次 pread（同 B-1）。
        // 不更新 stats_：并发累加不安全，且这条路径不走缓存。
        bool load_expert_direct(int layer, int expert, uint8_t *buf, size_t buf_nbytes,
                                ExpertWeights *out);

        // ---- B-2：异步预取 ----
        // 设计要点：预取用**独立缓冲池**，不复用 LRU 槽。这同时规避两个风险：
        //   ① 线程安全——预取线程只碰 pf_slots_/pf_queue_（mutex 保护），
        //      slots_/slot_index_ 仍只有主线程碰，无需给 LRU 加锁。
        //   ② 指针生命周期——预取槽是专用缓冲、不进 LRU，不会被淘汰。
        //      B-1 后三个指针指向同一 span 缓冲，若复用 LRU 槽则淘汰即全部失效。
        // 内存代价：pf_slots 个 × 单专家字节数（默认 8 × 2.5 MB = 20 MB），
        // 须计入启动内存预算校验。

        // 启用预取：分配 n 个专用缓冲槽并启动后台线程。必须在 register_expert
        // 之后调用（需要单专家字节数）。n 应 ≥ experts_per_tok，否则本层内会
        // 发生槽复用、先预取的被覆盖。
        bool prefetch_enable(int n, std::string *err);

        // 入队一个专家的预取请求（非阻塞）。通常在 topk_softmax 之后一次性
        // 入队本层全部 k 个专家，使 pread 与后续 expert_ffn 计算重叠。
        void prefetch_enqueue(int layer, int expert);

        // 预取缓冲池占用的字节数（供内存预算归因）
        uint64_t prefetch_pool_bytes() const { return pf_pool_bytes_; }

        // 关闭预取线程（析构也会调用）
        void prefetch_disable();

        // 丢弃 page cache（benchmark 测 cold-load 用：pread 后 posix_fadvise DONTNEED）
        void drop_page_cache();

    private:
        struct ExpertLayout {
            uint64_t gate_off, gate_nbytes;
            uint64_t up_off, up_nbytes;
            uint64_t down_off, down_nbytes;
            int inter, hidden, group_size;
            // 三块在文件内连续时可合并成 1 次 pread（B-1 优化）。exporter 按
            // gate_proj → up_proj → down_proj 顺序写入同一专家，且卸载张量保留
            // 原始文件偏移，故三者连续（块间仅 64B 对齐填充）。
            // contiguous 为真时 span_nbytes = down_off+down_nbytes - gate_off，
            // up_rel/down_rel 是 up/down 在跨度内的相对偏移。
            bool contiguous = false;
            uint64_t span_nbytes = 0;
            uint64_t up_rel = 0, down_rel = 0;
        };
        struct Slot {
            int64_t key = -1;     // (layer, expert) 编码；-1 = 空槽
            uint64_t tick = 0;    // LRU 时戳
            std::vector<uint8_t> gate, up, down;
            // 三块连续时用单一缓冲一次 pread（B-1）；此时 gate/up/down 为空，
            // ExpertWeights 的三个指针指向 span 内的不同偏移。
            std::vector<uint8_t> span;
            ExpertWeights w;
        };

        // (layer, expert) → 唯一 key
        static int64_t make_key(int layer, int expert) {
            return (static_cast<int64_t>(layer) << 20) | static_cast<int64_t>(expert & 0xFFFFF);
        }

        // 从 fd 在 offset 处读 exactly nbytes 字节到 buf（循环 pread，处理短读）
        bool pread_all(uint64_t offset, size_t nbytes, std::vector<uint8_t> &buf);

        int fd_ = -1;                              // .tqwen 文件描述符（pread 用）
        std::unordered_map<int64_t, ExpertLayout> layout_map_;

        std::vector<Slot> slots_;
        std::unordered_map<int64_t, size_t> slot_index_;  // key → slots_ 下标
        int max_slots_ = 4;
        uint64_t tick_ = 0;
        uint64_t cache_budget_bytes_ = 0;  // 0 = 未设预算（按槽数限界）
        uint64_t per_expert_bytes_ = 0;    // gate+up+down 三块之和（首个注册专家）
        uint64_t expert_staging_bytes_ = 0; // 直读所需字节数（连续布局取跨度）
        ExpertStoreStats stats_;
        std::atomic<uint64_t> direct_bytes_read_{0};  // 并行直读路径的 pread 字节

        // ---- B-2 预取状态（仅预取线程与主线程经 mutex 访问）----
        struct PrefetchSlot {
            int64_t key = -1;            // -1 = 空闲
            std::vector<uint8_t> buf;    // 专用缓冲（不进 LRU，不会被淘汰）
            bool ready = false;          // pread 完成
            ExpertWeights w;
        };
        std::vector<PrefetchSlot> pf_slots_;
        std::deque<int64_t> pf_queue_;         // 待预取的 key（FIFO）
        std::mutex pf_mutex_;
        // **两个 CV 而非一个**：等待条件不同（预取线程等"队列有活"，主线程等
        // "某 key 就绪"）。共用一个 CV + notify_one 会死锁——notify_one 可能唤醒
        // 错误的等待者（主线程），它重查条件不满足又睡回去，预取线程永远得不到
        // 唤醒、永不处理队列，双方永久互等。这是实测踩到的坑。
        std::condition_variable pf_work_cv_;   // 通知预取线程：队列有新任务
        std::condition_variable pf_ready_cv_;  // 通知主线程：某专家 pread 完成
        std::thread pf_thread_;
        bool pf_running_ = false;
        uint64_t pf_pool_bytes_ = 0;

        void prefetch_worker();                // 预取线程主循环
        // 在 pf_slots_ 里找 key 的槽；找不到返回 nullptr。调用方须持锁。
        PrefetchSlot *pf_find_locked(int64_t key);
        // 挑一个可复用的预取槽：空闲 → 已就绪，**绝不选在飞槽**。
        // 全在飞时 *ok=false，调用方应放弃本次预取。调用方须持锁。
        size_t pf_pick_slot_locked(bool *ok);
    };

} // namespace tinyqwen
