// Result = the output of one inference, delivered in FIFO order.
#pragma once

#include "dxinfer/buffer_pool.hpp"
#include "dxinfer/error.hpp"
#include "dxinfer/stats.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dxinfer {

namespace detail { struct ResultAccess; }

/// One output tensor of a Result. `data` points into the Result's own memory.
struct TensorView {
    std::string name;
    int dtype = 0;                ///< dxrt::DataType value
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;

    template <typename T>
    const T* as() const { return reinterpret_cast<const T*>(data); }
};

/// Move-only. A Result from the library owns one pool block (written directly
/// by dx-runtime); the block returns to the pool when the Result is destroyed.
/// Holding many pool-backed Results starves the device - keep long-lived
/// outputs with copy_out(), which owns a heap copy and holds no block.
class Result {
public:
    Result() = default;
    Result(Result&&) = default;
    Result& operator=(Result&&) = default;
    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    const std::string& network() const
    {
        static const std::string kNone;
        return network_ ? *network_ : kNone;
    }
    uint64_t seq() const noexcept { return seq_; }          ///< value feed() returned
    void* user_arg() const noexcept { return user_arg_; }   ///< value given to feed()
    const Status& status() const noexcept { return status_; }
    bool ok() const noexcept { return status_.ok(); }

    const std::vector<TensorView>& tensors() const noexcept { return tensors_; }
    size_t tensor_count() const noexcept { return tensors_.size(); }
    const TensorView& tensor(size_t i) const { return tensors_.at(i); }
    const TensorView& tensor(const std::string& tensor_name) const
    {
        for (const TensorView& t : tensors_)
            if (t.name == tensor_name) return t;
        throw std::out_of_range("Result: no output tensor named '" + tensor_name + "'");
    }

    /// Whole output memory (pool block or owned copy).
    const uint8_t* data() const noexcept
    {
        if (buffer_) return buffer_.data();
        return owned_.empty() ? nullptr : owned_.data();
    }
    uint64_t bytes() const noexcept { return bytes_; }
    const JobTimes& times() const noexcept { return times_; }  ///< filled only with stats enabled

    /// true: holds a block of the network's result pool; false: owns its memory (or is empty).
    bool uses_pool() const noexcept { return static_cast<bool>(buffer_); }

    /// A copy with the same metadata whose bytes live in its own heap memory.
    /// Keeping it never uses a pool block. Costs one copy of the output.
    Result copy_out() const;

private:
    friend struct detail::ResultAccess;

    size_t storage_size() const noexcept { return buffer_ ? buffer_.size() : owned_.size(); }

    std::shared_ptr<const std::string> network_;
    uint64_t seq_ = 0;
    void* user_arg_ = nullptr;
    Status status_;
    PooledBuffer buffer_;           // pool-backed Result
    std::vector<uint8_t> owned_;    // copy_out() Result (heap memory survives moves)
    std::vector<TensorView> tensors_;
    uint64_t bytes_ = 0;
    JobTimes times_;
};

namespace detail {

/// Internal: how the library (and the host unit tests) build a Result.
struct ResultAccess {
    static Result make(std::shared_ptr<const std::string> network, uint64_t seq, void* user_arg, Status status,
                       PooledBuffer buffer, std::vector<TensorView> tensors, uint64_t bytes, const JobTimes& times)
    {
        Result r;
        r.network_ = std::move(network);
        r.seq_ = seq;
        r.user_arg_ = user_arg;
        r.status_ = std::move(status);
        r.buffer_ = std::move(buffer);
        r.tensors_ = std::move(tensors);
        r.bytes_ = bytes;
        r.times_ = times;
        return r;
    }
};

}  // namespace detail

inline Result Result::copy_out() const
{
    Result c;
    c.network_ = network_;
    c.seq_ = seq_;
    c.user_arg_ = user_arg_;
    c.status_ = status_;
    c.bytes_ = bytes_;
    c.times_ = times_;

    const uint8_t* src = data();
    const size_t cap = storage_size();
    const uintptr_t lo = reinterpret_cast<uintptr_t>(src);
    size_t span = static_cast<size_t>(bytes_);  // copy up to the end of the last tensor, within the storage
    for (const TensorView& t : tensors_) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(t.data);
        if (src && t.data && a >= lo && a - lo < cap) span = std::max(span, static_cast<size_t>(a - lo + t.bytes));
    }
    span = std::min(span, cap);
    if (src && span > 0) c.owned_.assign(src, src + span);

    c.tensors_ = tensors_;
    for (TensorView& t : c.tensors_) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(t.data);
        const bool inside = src && t.data && a >= lo && a - lo + t.bytes <= span;
        t.data = inside ? c.owned_.data() + (a - lo) : nullptr;
    }
    return c;
}

}  // namespace dxinfer