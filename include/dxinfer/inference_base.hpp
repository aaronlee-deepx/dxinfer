// InferenceBase - the part shared by BlockingInference and CallbackInference:
// loads every network (all-or-nothing), answers network queries, feeds inputs
// and exposes statistics. How results are handed back is up to the subclass.
#pragma once

#include "dxinfer/error.hpp"
#include "dxinfer/stats.hpp"
#include "dxinfer/types.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dxinfer {

namespace detail { class Network; }

class InferenceBase {
public:
    virtual ~InferenceBase();
    InferenceBase(const InferenceBase&) = delete;
    InferenceBase& operator=(const InferenceBase&) = delete;

    // ---- network queries ---------------------------------------------------
    std::vector<std::string> network_names() const;  ///< in construction order
    bool has_network(const std::string& name) const;
    const NetworkInfo& info(const std::string& name) const;
    uint64_t input_size(const std::string& name) const;   ///< bytes feed() expects
    uint64_t output_size(const std::string& name) const;  ///< bytes of one result block
    const std::vector<TensorInfo>& input_tensors(const std::string& name) const;
    const std::vector<TensorInfo>& output_tensors(const std::string& name) const;
    void print_info(const std::string& name, std::ostream& os = std::cout) const;
    void print_info(std::ostream& os = std::cout) const;  ///< all networks
    ExecMode exec_mode() const noexcept { return cfg_.exec; }

    // ---- feeding -------------------------------------------------------------
    /// Submit one input of input_size(name) bytes (multi-input models: all input
    /// tensors back to back, layout in input_tensors()). Blocks only while every
    /// device slot of this network is busy. The input may be reused as soon as
    /// feed() returns. Returns the sequence number; this network's results are
    /// delivered in exactly this order.
    uint64_t feed(const std::string& name, const void* input, size_t bytes, void* user_arg = nullptr);

    /// Like feed() but never blocks: false if no slot / result block is free now.
    bool try_feed(const std::string& name, const void* input, size_t bytes, void* user_arg = nullptr,
                  uint64_t* seq = nullptr);

    /// Block until every job fed so far has finished on the device
    /// (its result may still be waiting to be consumed).
    void wait_completed();

    /// Jobs of `name` fed but not yet consumed by you.
    uint64_t pending(const std::string& name) const;

    // ---- statistics (InferenceConfig::enable_stats) -------------------------
    bool stats_enabled() const noexcept { return cfg_.enable_stats; }
    NetworkStats stats(const std::string& name) const;
    void reset_stats(const std::string& name);  ///< call while idle
    void reset_stats();
    void print_stats(const std::string& name, std::ostream& os = std::cout) const;
    void print_stats(std::ostream& os = std::cout) const;

protected:
    InferenceBase(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg);

    detail::Network& network(const std::string& name) const;  ///< throws UnknownNetwork
    const std::vector<std::unique_ptr<detail::Network>>& networks() const noexcept { return nets_; }

    /// Called by feed()/try_feed() before submitting; throw to refuse.
    virtual void check_can_feed(const detail::Network&) const {}

    /// Stop all networks: wait for in-flight jobs and close the result queues.
    /// Subclasses with their own threads call this first in their destructor.
    void close_all() noexcept;

private:
    InferenceConfig cfg_;
    std::vector<std::unique_ptr<detail::Network>> nets_;
    std::map<std::string, detail::Network*> by_name_;
};

}  // namespace dxinfer