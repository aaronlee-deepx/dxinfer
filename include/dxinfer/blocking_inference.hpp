// BlockingInference - results are pulled with get(), in FIFO order per network.
//
//   BlockingInference inf({{"det", "yolo.dxnn"}}, ExecMode::Async);
//   inf.feed("det", img, inf.input_size("det"));
//   Result r = inf.get("det");      // blocks until the next result is ready
//
// Feed and get from different threads to keep the device full. Results you
// have not taken yet hold result blocks; when all blocks are held, feed()
// waits (try_feed() returns false) until you get() some.
#pragma once

#include "dxinfer/inference_base.hpp"
#include "dxinfer/result.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace dxinfer {

class BlockingInference : public InferenceBase {
public:
    explicit BlockingInference(const std::vector<NetworkSpec>& specs, ExecMode exec = ExecMode::Async);
    BlockingInference(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg);

    /// Next result of `name` (FIFO). Blocks. Throws ShuttingDown once closed.
    Result get(const std::string& name);

    /// Never blocks: false if the next result is not ready yet.
    bool try_get(const std::string& name, Result& out);

    /// Waits up to `timeout`.
    template <class Rep, class Period>
    bool get_for(const std::string& name, const std::chrono::duration<Rep, Period>& timeout, Result& out)
    {
        return get_for_us(name, std::chrono::duration_cast<std::chrono::microseconds>(timeout), out);
    }

private:
    bool get_for_us(const std::string& name, std::chrono::microseconds timeout, Result& out);
};

}  // namespace dxinfer