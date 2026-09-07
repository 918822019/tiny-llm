// ============================================================================
// expert_store.cpp — MoE 路由专家 SSD 卸载存储实现
// ============================================================================
// 详见 expert_store.h 的设计说明。本文件实现：
//   - open(): 以 O_RDONLY 打开 .tqwen 文件（pread 用，不 mmap）
//   - register_expert(): 建 (layer,expert) → 三块 offset/nbytes 的表
//   - get(): LRU 查找；命中返指针；未命中淘汰 + pread 三块
//   - pread_all(): 循环 pread 处理短读，读满 nbytes
//   - drop_page_cache(): posix_fadvise DONTNEED 丢弃 page cache（benchmark 用）
// ============================================================================

#include "expert_store.h"

#include <limits>     // std::numeric_limits
#include <cstdio>    // fprintf
#include <cstdlib>   // abort
#include <fcntl.h>    // open, O_RDONLY
#include <unistd.h>   // pread, close, posix_fadvise

namespace tinyqwen {

    ExpertStore::~ExpertStore() {
        if (fd_ >= 0) ::close(fd_);
    }

    bool ExpertStore::open(const std::string &path, std::string *err) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            if (err) *err = "ExpertStore: cannot open '" + path + "'";
            return false;
        }
        return true;
    }

    void ExpertStore::register_expert(int layer, int expert,
                                       uint64_t gate_off, uint64_t gate_nbytes,
                                       uint64_t up_off, uint64_t up_nbytes,
                                       uint64_t down_off, uint64_t down_nbytes,
                                       int inter, int hidden, int group_size) {
        ExpertLayout L;
        L.gate_off = gate_off;     L.gate_nbytes = gate_nbytes;
        L.up_off = up_off;         L.up_nbytes = up_nbytes;
        L.down_off = down_off;     L.down_nbytes = down_nbytes;
        L.inter = inter;          L.hidden = hidden;   L.group_size = group_size;
        // 检测三块是否在文件内连续（允许块间 64B 对齐填充）。连续则 get() 可
        // 合并成 1 次 pread：syscall 从 3 次降到 1 次，且单次大顺序读更容易打满
        // NVMe 带宽（实测 4.45 GB/s = 峰值 74%，仍有提升空间）。
        // 判据：up 紧跟 gate、down 紧跟 up，间隙不超过 64 字节对齐填充。
        const uint64_t gate_end = gate_off + gate_nbytes;
        const uint64_t up_end = up_off + up_nbytes;
        const bool ordered = up_off >= gate_end && down_off >= up_end &&
                             (up_off - gate_end) <= 64 && (down_off - up_end) <= 64;
        if (ordered) {
            L.contiguous = true;
            L.span_nbytes = (down_off + down_nbytes) - gate_off;
            L.up_rel = up_off - gate_off;
            L.down_rel = down_off - gate_off;
        }
        layout_map_[make_key(layer, expert)] = L;
        // 首个注册专家决定单专家字节数（同模型所有专家同形状）。字节预算要靠它
        // 把"预算 MB"换算成"槽数"，所以必须在 register_expert 之后才能设预算。
        if (per_expert_bytes_ == 0) {
            per_expert_bytes_ = gate_nbytes + up_nbytes + down_nbytes;
        }
    }

    bool ExpertStore::set_cache_budget(uint64_t budget_bytes, std::string *err) {
        if (per_expert_bytes_ == 0) {
            if (err) *err = "set_cache_budget 必须在 register_expert() 之后调用"
                            "（需要单专家字节数才能换算槽数）";
            return false;
        }
        // 单个专家都装不下 → fail-fast。静默降级成 slots=0 会让用户误以为预算生效，
        // 而实际每次访问都 pread（行为完全不同），这是本仓库最忌讳的失效模式。
        if (budget_bytes < per_expert_bytes_) {
            if (err) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "专家缓存预算 %llu B 小于单个专家 %llu B，装不下任何专家",
                              static_cast<unsigned long long>(budget_bytes),
                              static_cast<unsigned long long>(per_expert_bytes_));
                *err = buf;
            }
            return false;
        }
        cache_budget_bytes_ = budget_bytes;
        set_cache_slots(static_cast<int>(budget_bytes / per_expert_bytes_));
        return true;
    }

    void ExpertStore::set_cache_slots(int n) {
        if (n < 0) n = 0;
        max_slots_ = n;
        slots_.clear();
        slots_.resize(max_slots_);
        slot_index_.clear();
    }

    bool ExpertStore::pread_all(uint64_t offset, size_t nbytes,
                                std::vector<uint8_t> &buf) {
        if (buf.size() < nbytes) buf.resize(nbytes);
        size_t done = 0;
        while (done < nbytes) {
            ssize_t r = ::pread(fd_, buf.data() + done, nbytes - done,
                                static_cast<off_t>(offset + done));
            if (r < 0) {
                std::fprintf(stderr,
                             "tinyqwen: ExpertStore pread failed at off=%llu nbytes=%zu\n",
                             static_cast<unsigned long long>(offset + done), nbytes - done);
                return false;
            }
            if (r == 0) break;  // EOF（不该发生，offset/nbytes 已校验）
            done += static_cast<size_t>(r);
        }
        return done == nbytes;
    }

    bool ExpertStore::read_bytes(uint64_t offset, size_t nbytes, void *out) {
        uint8_t *dst = static_cast<uint8_t *>(out);
        size_t done = 0;
        while (done < nbytes) {
            ssize_t r = ::pread(fd_, dst + done, nbytes - done,
                                static_cast<off_t>(offset + done));
            if (r < 0) {
                std::fprintf(stderr,
                             "tinyqwen: ExpertStore read_bytes failed at off=%llu "
                             "nbytes=%zu\n",
                             static_cast<unsigned long long>(offset + done), nbytes - done);
                return false;
            }
            if (r == 0) break;
            done += static_cast<size_t>(r);
        }
        return done == nbytes;
    }

    const ExpertWeights ExpertStore::get(int layer, int expert) {
        const int64_t key = make_key(layer, expert);
        auto lit = layout_map_.find(key);
        if (lit == layout_map_.end()) {
            std::fprintf(stderr, "tinyqwen: ExpertStore: expert (layer=%d,expert=%d) 未注册\n",
                         layer, expert);
            std::abort();
        }
        const ExpertLayout &L = lit->second;

        // ---- 命中缓存 ----
        auto sit = slot_index_.find(key);
        if (sit != slot_index_.end()) {
            Slot &s = slots_[sit->second];
            s.tick = ++tick_;
            ++stats_.hits;
            return s.w;
        }

        // ---- 未命中：挑一个槽（空槽或 LRU）----
        ++stats_.misses;
        size_t slot_idx;
        if (slots_.empty()) {
            // max_slots_ == 0：每次都现读现返，无缓存。用一个临时静态缓冲。
            // 注意：返回的指针在下一次 get 调用前有效（单线程 decode 内安全）。
            static thread_local std::vector<uint8_t> t_gate, t_up, t_down, t_span;
            if (L.contiguous) {
                if (!pread_all(L.gate_off, L.span_nbytes, t_span)) std::abort();
                stats_.bytes_read += L.span_nbytes;
                return ExpertWeights{t_span.data(), t_span.data() + L.up_rel,
                                     t_span.data() + L.down_rel};
            }
            if (!pread_all(L.gate_off, L.gate_nbytes, t_gate) ||
                !pread_all(L.up_off, L.up_nbytes, t_up) ||
                !pread_all(L.down_off, L.down_nbytes, t_down)) {
                std::abort();
            }
            stats_.bytes_read += L.gate_nbytes + L.up_nbytes + L.down_nbytes;
            return ExpertWeights{t_gate.data(), t_up.data(), t_down.data()};
        }
        // 找空槽或最小 tick
        slot_idx = 0;
        uint64_t min_tick = std::numeric_limits<uint64_t>::max();
        bool found_empty = false;
        for (size_t i = 0; i < slots_.size(); ++i) {
            if (slots_[i].key < 0) { slot_idx = i; found_empty = true; break; }
            if (slots_[i].tick < min_tick) { min_tick = slots_[i].tick; slot_idx = i; }
        }
        Slot &s = slots_[slot_idx];
        if (!found_empty && s.key >= 0) {
            // 淘汰旧占用者
            slot_index_.erase(s.key);
            ++stats_.evictions;
        }
        // 读三块进槽。连续时合并成 1 次 pread（B-1）：syscall 3→1，且单次大
        // 顺序读更容易打满 NVMe 带宽。
        if (L.contiguous) {
            if (!pread_all(L.gate_off, L.span_nbytes, s.span)) std::abort();
            s.gate.clear(); s.up.clear(); s.down.clear();
            stats_.bytes_read += L.span_nbytes;
            s.w.gate = s.span.data();
            s.w.up = s.span.data() + L.up_rel;
            s.w.down = s.span.data() + L.down_rel;
        } else {
            if (!pread_all(L.gate_off, L.gate_nbytes, s.gate) ||
                !pread_all(L.up_off, L.up_nbytes, s.up) ||
                !pread_all(L.down_off, L.down_nbytes, s.down)) {
                std::abort();
            }
            s.span.clear();
            stats_.bytes_read += L.gate_nbytes + L.up_nbytes + L.down_nbytes;
            s.w.gate = s.gate.data();
            s.w.up = s.up.data();
            s.w.down = s.down.data();
        }
        s.key = key;
        s.tick = ++tick_;
        slot_index_[key] = slot_idx;
        return s.w;
    }

    void ExpertStore::drop_page_cache() {
        if (fd_ < 0) return;
#ifdef POSIX_FADV_DONTNEED
        ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
#endif
    }

} // namespace tinyqwen
