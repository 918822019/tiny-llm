#include "expert_pool.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <cstdlib>
#include <cstdio>

namespace tinyqwen {

    namespace {
        inline void spin_until(const std::atomic<std::uint64_t> &a, std::uint64_t target) {
            int spins = 0;
            while (a.load(std::memory_order_acquire) != target) {
                if (++spins <= 256) {
#if defined(__aarch64__)
                    __builtin_arm_yield();
#endif
                } else {
                    std::this_thread::yield();
                }
            }
        }
    } // namespace

    ExpertPool::ExpertPool(int threads) {
        if (threads > 16) threads = 16;
        if (threads < 2) return;   // 串行：不建 worker，run() 内联执行
        workers_.reserve(static_cast<size_t>(threads - 1));
        for (int idx = 1; idx < threads; ++idx) {
            workers_.emplace_back([this, idx] { worker_main(idx); });
        }
    }

    ExpertPool::~ExpertPool() {
        shutdown_.store(true, std::memory_order_release);
        job_gen_.fetch_add(1, std::memory_order_release);
        for (auto &t : workers_) t.join();
    }

    void ExpertPool::worker_main(int idx) {
        std::uint64_t next_job = 1;
        for (;;) {
            spin_until(job_gen_, next_job);
            if (shutdown_.load(std::memory_order_acquire)) return;
            do_chunk(idx);
            done_gen_.fetch_add(1, std::memory_order_release);
            ++next_job;
        }
    }

    void ExpertPool::do_chunk(int idx) const {
        const int p = static_cast<int>(workers_.size()) + 1;
        const int base = n_tasks_ / p;
        const int rem = n_tasks_ % p;
        const int begin = idx * base + (idx < rem ? idx : rem);
        const int end = begin + base + (idx < rem ? 1 : 0);
        for (int i = begin; i < end; ++i) (*task_fn_)(i);
    }

    void ExpertPool::run(const TaskFn &fn, int n) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        task_fn_ = &fn;
        n_tasks_ = n;
        job_gen_.store(++job_counter_, std::memory_order_release);
        do_chunk(0);
        expected_done_ += workers_.size();
        spin_until(done_gen_, expected_done_);
        task_fn_ = nullptr;
    }

} // namespace tinyqwen
