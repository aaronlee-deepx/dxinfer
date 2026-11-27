// detail::Network - one loaded network (internal to the library).
//
// Owns the dx-runtime engine and everything needed to feed it at full speed
// and give results back in FIFO order:
//   job slots     : at most max_in_flight jobs (dxrt buffers x devices) on the
//                   device; a slot is freed the moment its job completes, so a
//                   waiting feed() refills the device immediately
//   output pool   : zero-copy result blocks (dx-runtime writes into them)
//   reorder buffer: completions in any order -> results in seq order
//   stats         : optional, no-op hooks when disabled
#pragma once

#include "dxinfer/buffer_pool.hpp"
#include "dxinfer/reorder_buffer.hpp"
#include "dxinfer/result.hpp"
#include "dxinfer/stats.hpp"
#include "dxinfer/types.hpp"

#include <dxrt/dxrt_cxx_api.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dxinfer {
namespace detail {

class Network {
public:
    Network(const NetworkSpec& spec, const InferenceConfig& cfg);  // throws InferenceError
    ~Network();
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;

    const std::string& name() const noexcept { return *name_; }
    const NetworkInfo& info() const noexcept { return info_; }

    /// Submit one input. wait=false: return false instead of blocking when no
    /// slot or result block is free. Throws on misuse / shutdown only.
    bool submit(const void* input, size_t bytes, void* user_arg, bool wait, uint64_t* seq_out);

    ReorderBuffer<Result>& results() noexcept { return results_; }
    void on_delivered(const Result& r)  ///< handed to the user (popped from results())
    {
        occ_->waiting.fetch_sub(1, std::memory_order_relaxed);
        stats_.on_deliver(r.times());
    }
    void on_consumed();                                                   ///< user is done with it

    void wait_completed();  ///< every fed job finished on the device
    void wait_consumed();   ///< every fed job consumed by the user
    uint64_t pending() const;

    NetworkStats stats() const;
    void reset_stats() { stats_.reset(); }

    /// Stop accepting feeds, wake blocked feeders, wait for in-flight jobs,
    /// close the result queue. Idempotent.
    void close();

    /// Where the result blocks are, for the low-buffer warning. Shared (not `this`) so a
    /// Result dropped late can still run the pool observer safely.
    struct Occupancy {
        std::atomic<size_t> in_flight{0};  ///< jobs holding a slot (and a block)
        std::atomic<size_t> waiting{0};    ///< completed, not yet popped from the reorder buffer
    };

private:
    struct Job {
        uint64_t seq = 0;
        void* user_arg = nullptr;
        PooledBuffer buffer;
        JobTimes times;
    };

    void build_info(const NetworkSpec& spec, const InferenceConfig& cfg);
    Job* acquire_job(bool wait, int* in_flight);
    void release_job(Job* job);
    std::vector<void*> split_input(void* input) const;
    void complete(Job* job, const dxrt::TensorPtrs* outputs, Status status);  // any thread
    std::vector<TensorView> make_views(uint8_t* base, size_t capacity, const dxrt::TensorPtrs* outputs,
                                       Status& status) const;

    std::shared_ptr<const std::string> name_;
    NetworkInfo info_;
    ExecMode exec_;
    std::vector<uint64_t> input_parts_;  // multi-input models: byte size of each input tensor
    std::unique_ptr<dxrt::InferenceEngine> engine_;
    std::shared_ptr<BufferPool> pool_;

    std::vector<Job> jobs_;  // fixed size = max_in_flight; never resized (pointers are passed to dxrt)
    std::mutex slot_m_;
    std::condition_variable slot_cv_;
    std::vector<Job*> free_jobs_;
    bool closing_ = false;

    std::atomic<uint64_t> next_seq_{0};
    ReorderBuffer<Result> results_;

    mutable std::mutex count_m_;
    std::condition_variable count_cv_;
    uint64_t fed_ = 0, completed_ = 0, consumed_ = 0;

    StatsRecorder stats_;
    std::shared_ptr<Occupancy> occ_ = std::make_shared<Occupancy>();
};

}  // namespace detail
}  // namespace dxinfer