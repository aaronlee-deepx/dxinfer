// Fixed-size block pool used for zero-copy outputs.
//
// dx-runtime writes each inference result straight into one block (the block
// is passed as the output pointer of Run/RunAsync). The block travels inside
// the Result and goes back to the pool when the Result is destroyed (RAII).
// A PooledBuffer keeps its pool alive, so a Result may safely outlive the
// inference object that produced it.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace dxinfer {

class BufferPool;

/// Move-only handle to one pool block. Empty (false) when default-constructed
/// or when the pool was shut down.
class PooledBuffer {
public:
    PooledBuffer() = default;
    PooledBuffer(PooledBuffer&& o) noexcept : pool_(std::move(o.pool_)), data_(o.data_) { o.data_ = nullptr; }
    PooledBuffer& operator=(PooledBuffer&& o) noexcept
    {
        if (this != &o) {
            reset();
            pool_ = std::move(o.pool_);
            data_ = o.data_;
            o.data_ = nullptr;
        }
        return *this;
    }
    PooledBuffer(const PooledBuffer&) = delete;
    PooledBuffer& operator=(const PooledBuffer&) = delete;
    ~PooledBuffer() { reset(); }

    uint8_t* data() const noexcept { return data_; }
    size_t size() const noexcept;
    explicit operator bool() const noexcept { return data_ != nullptr; }
    void reset() noexcept;  ///< give the block back now

private:
    friend class BufferPool;
    PooledBuffer(std::shared_ptr<BufferPool> pool, uint8_t* data) : pool_(std::move(pool)), data_(data) {}

    std::shared_ptr<BufferPool> pool_;
    uint8_t* data_ = nullptr;
};

class BufferPool : public std::enable_shared_from_this<BufferPool> {
public:
    static constexpr size_t kAlignment = 64;

    /// Level observer: fn(true, free, total) when a take leaves free <= low_mark
    /// (once per episode); fn(false, free, total) when a return brings
    /// free >= recover_mark. Called outside the pool lock, on the thread that
    /// took / returned the block. Must be quick.
    using LevelFn = std::function<void(bool low, size_t free_blocks, size_t total_blocks)>;

    static std::shared_ptr<BufferPool> create(size_t block_bytes, size_t block_count)
    {
        return std::shared_ptr<BufferPool>(new BufferPool(block_bytes, block_count));
    }

    /// Blocks until a block is free. Returns an empty handle after shutdown().
    PooledBuffer acquire()
    {
        PooledBuffer b;
        std::shared_ptr<const LevelFn> fire;
        size_t free_now = 0;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this] { return !free_.empty() || stopped_; });
            b = take_locked();
            if (b) fire = went_low_locked(&free_now);
        }
        notify(fire, true, free_now);
        return b;
    }

    /// Never blocks. Empty handle if no block is free.
    PooledBuffer try_acquire()
    {
        PooledBuffer b;
        std::shared_ptr<const LevelFn> fire;
        size_t free_now = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            b = take_locked();
            if (b) fire = went_low_locked(&free_now);
        }
        notify(fire, true, free_now);
        return b;
    }

    /// Install (or with fn == nullptr remove) the level observer.
    void set_level_watch(size_t low_mark, size_t recover_mark, std::shared_ptr<const LevelFn> fn)
    {
        std::lock_guard<std::mutex> lk(m_);
        low_mark_ = low_mark;
        recover_mark_ = recover_mark;
        watch_ = std::move(fn);
        low_ = false;
    }

    /// Wake all waiters; afterwards acquire()/try_acquire() return empty handles.
    /// Blocks already handed out can still be returned.
    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lk(m_);
            stopped_ = true;
        }
        cv_.notify_all();
    }

    size_t block_bytes() const noexcept { return block_bytes_; }
    size_t capacity() const noexcept { return count_; }
    size_t available() const
    {
        std::lock_guard<std::mutex> lk(m_);
        return free_.size();
    }

private:
    friend class PooledBuffer;

    BufferPool(size_t block_bytes, size_t block_count)
        : block_bytes_(block_bytes),
          stride_(round_up(block_bytes == 0 ? 1 : block_bytes)),
          count_(block_count),
          storage_(new uint8_t[stride_ * block_count + kAlignment])
    {
        uint8_t* base = align(storage_.get());
        free_.reserve(count_);
        for (size_t i = 0; i < count_; ++i) free_.push_back(base + i * stride_);
    }

    static size_t round_up(size_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }
    static uint8_t* align(uint8_t* p)
    {
        const uintptr_t v = (reinterpret_cast<uintptr_t>(p) + kAlignment - 1) & ~static_cast<uintptr_t>(kAlignment - 1);
        return reinterpret_cast<uint8_t*>(v);
    }

    PooledBuffer take_locked()
    {
        if (stopped_ || free_.empty()) return PooledBuffer();
        uint8_t* p = free_.back();
        free_.pop_back();
        return PooledBuffer(shared_from_this(), p);
    }

    void give_back(uint8_t* p)
    {
        std::shared_ptr<const LevelFn> fire;
        size_t free_now = 0;
        {
            std::lock_guard<std::mutex> lk(m_);
            free_.push_back(p);
            if (watch_ && low_ && free_.size() >= recover_mark_) {
                low_ = false;
                free_now = free_.size();
                fire = watch_;
            }
        }
        cv_.notify_one();
        notify(fire, false, free_now);
    }

    std::shared_ptr<const LevelFn> went_low_locked(size_t* free_now)
    {
        if (!watch_ || low_ || free_.size() > low_mark_) return nullptr;
        low_ = true;
        *free_now = free_.size();
        return watch_;
    }

    void notify(const std::shared_ptr<const LevelFn>& fn, bool low, size_t free_now) const noexcept
    {
        if (!fn || !*fn) return;
        try {
            (*fn)(low, free_now, count_);
        } catch (...) {
            // an observer must never break buffer accounting
        }
    }

    const size_t block_bytes_;
    const size_t stride_;
    const size_t count_;
    std::unique_ptr<uint8_t[]> storage_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::vector<uint8_t*> free_;
    bool stopped_ = false;
    std::shared_ptr<const LevelFn> watch_;
    size_t low_mark_ = 0;
    size_t recover_mark_ = 0;
    bool low_ = false;
};

inline size_t PooledBuffer::size() const noexcept { return pool_ ? pool_->block_bytes() : 0; }

inline void PooledBuffer::reset() noexcept
{
    if (data_ && pool_) pool_->give_back(data_);
    data_ = nullptr;
    pool_.reset();
}

}  // namespace dxinfer