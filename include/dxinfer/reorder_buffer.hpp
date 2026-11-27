// ReorderBuffer<T> restores FIFO order.
//
// With several jobs in flight and several NPU cores, dx-runtime completes jobs
// in any order. Each job carries the sequence number it got at feed(); the
// completion is push()ed here, and consumers only ever receive the item whose
// sequence number is next. Generic and dx-runtime-free (unit-tested on host).
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace dxinfer {

template <typename T>
class ReorderBuffer {
public:
    explicit ReorderBuffer(uint64_t first_seq = 0) : next_(first_seq) {}

    /// Store the item of `seq`. Any thread, any order; each seq exactly once.
    void push(uint64_t seq, T item)
    {
        bool wake = false;
        {
            std::lock_guard<std::mutex> lk(m_);
            if (seq < next_ || pending_.count(seq) != 0)
                throw std::logic_error("ReorderBuffer: sequence number pushed twice");
            pending_.emplace(seq, std::move(item));
            wake = (seq == next_);
        }
        if (wake) cv_.notify_all();
    }

    /// Block until the next in-order item is ready. false = closed and nothing ready.
    bool pop(T& out)
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [this] { return head_ready_locked() || closed_; });
        return take_locked(out);
    }

    /// Never blocks.
    bool try_pop(T& out)
    {
        std::lock_guard<std::mutex> lk(m_);
        return take_locked(out);
    }

    /// Block up to `timeout`.
    template <class Rep, class Period>
    bool pop_for(T& out, const std::chrono::duration<Rep, Period>& timeout)
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait_for(lk, timeout, [this] { return head_ready_locked() || closed_; });
        return take_locked(out);
    }

    /// Wake all consumers. Ready items can still be popped; then pop() returns false.
    void close()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return closed_;
    }

    /// Items waiting (ready or held back by a missing earlier seq).
    size_t size() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return pending_.size();
    }

    uint64_t next_seq() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return next_;
    }

private:
    bool head_ready_locked() const { return !pending_.empty() && pending_.begin()->first == next_; }

    bool take_locked(T& out)
    {
        if (!head_ready_locked()) return false;
        auto it = pending_.begin();
        out = std::move(it->second);
        pending_.erase(it);
        ++next_;
        if (head_ready_locked()) cv_.notify_one();  // the next one is already here
        return true;
    }

    mutable std::mutex m_;
    std::condition_variable cv_;
    std::map<uint64_t, T> pending_;
    uint64_t next_;
    bool closed_ = false;
};

}  // namespace dxinfer