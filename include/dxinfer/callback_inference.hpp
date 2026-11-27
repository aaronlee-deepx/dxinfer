// CallbackInference - results are pushed to your callback, in FIFO order per network.
//
//   CallbackInference inf({{"det", "yolo.dxnn"}}, ExecMode::Async);
//   inf.set_callback("det", [](Result& r) { ... });
//   inf.feed("det", img, inf.input_size("det"));
//   inf.wait_idle();
//
// Each network has its own delivery thread: callbacks of one network run one
// at a time in seq order; different networks run in parallel. Callbacks never
// run on a dx-runtime thread, so slow user code cannot stall the NPU (it only
// delays later results of the same network). The Result's block returns to
// the pool when the callback returns, unless you std::move the Result out.
#pragma once

#include "dxinfer/inference_base.hpp"
#include "dxinfer/result.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dxinfer {

using ResultCallback = std::function<void(Result&)>;

class CallbackInference : public InferenceBase {
public:
    explicit CallbackInference(const std::vector<NetworkSpec>& specs, ExecMode exec = ExecMode::Async);
    CallbackInference(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg);
    /// Waits for in-flight jobs and delivers every remaining result before returning.
    ~CallbackInference() override;

    /// Set (or replace) the callback of one network. Must be set before its first feed().
    void set_callback(const std::string& name, ResultCallback cb);
    /// Same callback for every network (use Result::network() to tell them apart).
    void set_callback(ResultCallback cb);

    /// Block until every job fed so far has been delivered and its callback returned.
    void wait_idle();

    /// Exceptions thrown by the callback of `name` (caught, counted, printed to stderr).
    uint64_t callback_errors(const std::string& name) const;

protected:
    void check_can_feed(const detail::Network& n) const override;

private:
    struct Lane;
    Lane& lane(const std::string& name) const;
    void run(Lane& ln);
    void stop_lanes() noexcept;

    std::vector<std::unique_ptr<Lane>> lanes_;
    std::map<std::string, Lane*> lane_by_name_;
};

}  // namespace dxinfer