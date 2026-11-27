#include "dxinfer/pool_warning.hpp"

#include <sstream>
#include <utility>

namespace dxinfer {

namespace {
constexpr const char* kIndent = "          ";  // aligns continuation lines under "[dxinfer] "
}

PoolWarner::PoolWarner(std::string network, std::ostream& os, std::chrono::milliseconds min_interval,
                       int extra_result_buffers)
    : network_(std::move(network)), os_(&os), interval_(min_interval), extra_(extra_result_buffers)
{
}

void PoolWarner::on_level(bool low, size_t free_blocks, size_t total_blocks, const PoolOccupancy& occ)
{
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(m_);
    if (low) {
        ++low_events_;
        low_since_ = now;
        if (printed_any_ && now - last_print_ < interval_) {  // rate limit
            ++suppressed_;
            low_printed_ = false;
            return;
        }
        std::ostringstream m;  // one write per message, so lines from other threads cannot interleave
        m << "[dxinfer] WARNING " << network_ << ": result blocks low (" << free_blocks << " of " << total_blocks
          << " free)";
        if (suppressed_ > 0)
            m << " (" << suppressed_ << " similar warning" << (suppressed_ == 1 ? "" : "s") << " suppressed)";
        m << "\n";
        const std::string raise = extra_ >= 0
                                      ? "raise InferenceConfig::extra_result_buffers (now " + std::to_string(extra_) + ")"
                                      : "raise InferenceConfig::extra_result_buffers";
        const bool known = occ.in_flight + occ.waiting + occ.held_by_app > 0;
        if (known)
            m << kIndent << "in flight " << occ.in_flight << " | waiting for FIFO order " << occ.waiting
              << " | held by your app " << occ.held_by_app << "\n";
        if (known && occ.waiting > 0 && occ.waiting >= occ.held_by_app) {
            m << kIndent << "Cause: an earlier job is slow, so later results wait for their turn. Usually start-up warm-up.\n"
              << kIndent << "If it persists, " << raise << ".\n";
        } else if (known && occ.held_by_app > 0) {
            m << kIndent << "Cause: your app is still holding " << occ.held_by_app << " Result"
              << (occ.held_by_app == 1 ? "" : "s") << ". Drop them sooner,\n"
              << kIndent << "or keep long-lived ones with Result::copy_out().\n";
        } else {
            m << kIndent << "feed() waits when none are left. Drop Results sooner, keep long-lived ones with\n"
              << kIndent << "Result::copy_out(), or " << raise << ".\n";
        }
        *os_ << m.str() << std::flush;
        suppressed_ = 0;
        printed_any_ = true;
        low_printed_ = true;
        last_print_ = now;
    } else if (low_printed_) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - low_since_).count();
        std::ostringstream m;
        m << "[dxinfer] " << network_ << ": result blocks recovered (" << free_blocks << " of " << total_blocks
          << " free) after " << ms << " ms\n";
        *os_ << m.str() << std::flush;
        low_printed_ = false;
    }
}

uint64_t PoolWarner::low_events() const
{
    std::lock_guard<std::mutex> lk(m_);
    return low_events_;
}

}  // namespace dxinfer