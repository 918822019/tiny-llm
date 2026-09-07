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
        layout_map_[make_key(layer, expert)] = L;
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
            static thread_local std::vector<uint8_t> t_gate, t_up, t_down;
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
        // 读三块进槽
        if (!pread_all(L.gate_off, L.gate_nbytes, s.gate) ||
            !pread_all(L.up_off, L.up_nbytes, s.up) ||
            !pread_all(L.down_off, L.down_nbytes, s.down)) {
            std::abort();
        }
        stats_.bytes_read += L.gate_nbytes + L.up_nbytes + L.down_nbytes;
        s.key = key;
        s.tick = ++tick_;
        s.w.gate = s.gate.data();
        s.w.up = s.up.data();
        s.w.down = s.down.data();
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
