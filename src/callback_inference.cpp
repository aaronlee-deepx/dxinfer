#include "dxinfer/callback_inference.hpp"

#include "network.hpp"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

namespace dxinfer {

/// One delivery lane per network: pops results in FIFO order, runs the callback.
struct CallbackInference::Lane {
    detail::Network* net = nullptr;
    mutable std::mutex m;
    std::shared_ptr<const ResultCallback> cb;  // swapped atomically under m
    std::atomic<uint64_t> errors{0};
    std::thread thread;
};

CallbackInference::CallbackInference(const std::vector<NetworkSpec>& specs, ExecMode exec)
    : CallbackInference(specs, config_for(exec))
{
}

CallbackInference::CallbackInference(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg)
    : InferenceBase(specs, cfg)
{
    try {
        for (const auto& n : networks()) {
            std::unique_ptr<Lane> ln(new Lane);
            ln->net = n.get();
            lane_by_name_[n->name()] = ln.get();
            lanes_.push_back(std::move(ln));
        }
        for (auto& ln : lanes_) {
            Lane* raw = ln.get();
            raw->thread = std::thread([this, raw] { run(*raw); });
        }
    } catch (...) {
        stop_lanes();
        throw;
    }
}

CallbackInference::~CallbackInference() { stop_lanes(); }

void CallbackInference::stop_lanes() noexcept
{
    close_all();  // in-flight jobs finish, queues close -> lanes deliver the rest and exit
    for (auto& ln : lanes_)
        if (ln->thread.joinable()) ln->thread.join();
}

void CallbackInference::run(Lane& ln)
{
    Result r;
    while (ln.net->results().pop(r)) {
        ln.net->on_delivered(r);
        std::shared_ptr<const ResultCallback> cb;
        {
            std::lock_guard<std::mutex> lk(ln.m);
            cb = ln.cb;
        }
        try {
            if (cb && *cb) (*cb)(r);
        } catch (const std::exception& e) {
            ++ln.errors;
            std::fprintf(stderr, "[dxinfer] %s: callback threw: %s\n", ln.net->name().c_str(), e.what());
        } catch (...) {
            ++ln.errors;
            std::fprintf(stderr, "[dxinfer] %s: callback threw a non-std exception\n", ln.net->name().c_str());
        }
        r = Result();  // give the block back before reporting the job as consumed
        ln.net->on_consumed();
    }
}

CallbackInference::Lane& CallbackInference::lane(const std::string& name) const
{
    network(name);  // throws UnknownNetwork with the list of known names
    return *lane_by_name_.at(name);
}

void CallbackInference::set_callback(const std::string& name, ResultCallback cb)
{
    Lane& ln = lane(name);
    auto shared = std::make_shared<const ResultCallback>(std::move(cb));
    std::lock_guard<std::mutex> lk(ln.m);
    ln.cb = std::move(shared);
}

void CallbackInference::set_callback(ResultCallback cb)
{
    auto shared = std::make_shared<const ResultCallback>(std::move(cb));
    for (auto& ln : lanes_) {
        std::lock_guard<std::mutex> lk(ln->m);
        ln->cb = shared;
    }
}

void CallbackInference::check_can_feed(const detail::Network& n) const
{
    Lane& ln = lane(n.name());
    std::lock_guard<std::mutex> lk(ln.m);
    if (!ln.cb || !*ln.cb)
        throw InferenceError(ErrorCode::NoCallback, n.name(), "call set_callback() before feed()");
}

void CallbackInference::wait_idle()
{
    for (const auto& n : networks()) n->wait_consumed();
}

uint64_t CallbackInference::callback_errors(const std::string& name) const { return lane(name).errors; }

}  // namespace dxinfer