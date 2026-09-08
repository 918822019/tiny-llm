#ifndef TINYQWEN_EXPERT_POOL_H
#define TINYQWEN_EXPERT_POOL_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace tinyqwen {

    // =========================================================================
    // ExpertPool — MoE 专家级并行的常驻线程池
    // =========================================================================
    // 为什么需要它而不是复用 kernel 级多线程：专家矩阵 [768,2048] 只有 12 个
    // o_block 可切，kernel 级 MT 负载不均且每 token 产生 1152 次 fork-join
    // （3 matvec × 8 专家 × 48 层），同步开销压过收益（AGENTS.md 坑 #22）。
    // 专家级并行的 top-k 任务彼此完全独立，每 token 只有 48 次 fork-join。
    //
    // 同步用 spin + 原子计数器而非 condition_variable：坑 #27(a) 实测过——
    // 不同等待条件共用一个 CV 时 notify_one 会唤醒错误的等待者，双方永久互等。
    class ExpertPool {
    public:
        // fn 的参数是任务下标 [0, n)。调用方须保证 fn 内部各任务写各自的目标，
        // 不得共享可变状态（本池不加锁）。
        using TaskFn = std::function<void(int)>;

        // threads < 2 时不建 worker，run() 退化为串行内联执行。
        explicit ExpertPool(int threads);
        ~ExpertPool();

        ExpertPool(const ExpertPool &) = delete;
        ExpertPool &operator=(const ExpertPool &) = delete;

        // 实际并行度（含 master）
        int threads() const { return static_cast<int>(workers_.size()) + 1; }

        // 提交 n 个独立任务并等全部完成（fork-join）。master 算第 0 块。
        void run(const TaskFn &fn, int n);

    private:
        void worker_main(int idx);
        void do_chunk(int idx) const;

        const std::function<void(int)> *task_fn_ = nullptr;  // 由 master 在发布前设置
        int n_tasks_ = 0;

        std::atomic<std::uint64_t> job_gen_{0};
        std::atomic<std::uint64_t> done_gen_{0};
        std::atomic<bool> shutdown_{false};

        std::vector<std::thread> workers_;
        std::uint64_t job_counter_ = 0;
        std::uint64_t expected_done_ = 0;
    };

} // namespace tinyqwen

#endif // TINYQWEN_EXPERT_POOL_H
