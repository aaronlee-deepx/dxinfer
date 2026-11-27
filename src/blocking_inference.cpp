#include "dxinfer/blocking_inference.hpp"

#include "network.hpp"

namespace dxinfer {

BlockingInference::BlockingInference(const std::vector<NetworkSpec>& specs, ExecMode exec)
    : BlockingInference(specs, config_for(exec))
{
}

BlockingInference::BlockingInference(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg)
    : InferenceBase(specs, cfg)
{
}

// A result counts as consumed as soon as it is handed to the caller.
static void hand_over(detail::Network& n, const Result& r)
{
    n.on_delivered(r);
    n.on_consumed();
}

Result BlockingInference::get(const std::string& name)
{
    detail::Network& n = network(name);
    Result r;
    if (!n.results().pop(r)) throw InferenceError(ErrorCode::ShuttingDown, name, "result queue is closed");
    hand_over(n, r);
    return r;
}

bool BlockingInference::try_get(const std::string& name, Result& out)
{
    detail::Network& n = network(name);
    if (!n.results().try_pop(out)) return false;
    hand_over(n, out);
    return true;
}

bool BlockingInference::get_for_us(const std::string& name, std::chrono::microseconds timeout, Result& out)
{
    detail::Network& n = network(name);
    if (!n.results().pop_for(out, timeout)) return false;
    hand_over(n, out);
    return true;
}

}  // namespace dxinfer