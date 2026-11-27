// PoolWarner - turns a network's result-pool level events into short stderr
// messages. Installed by the library on every network whose pool has at least
// 2 spare blocks; there is no application hook (stderr only, by design).
//
//   Low       : free blocks dropped to <= 1. The message says where the blocks are
//               (in flight / waiting for FIFO order / held by your app) and names
//               the main cause. At most one message per interval (default 1 s)
//               per network; skipped ones are counted and reported in the next one.
//   Recovered : free blocks back to >= 3. Printed only if its Low was printed.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>

namespace dxinfer {

/// Where a network's result blocks are when the pool runs low.
struct PoolOccupancy {
    size_t in_flight = 0;    ///< jobs on the device (each owns a block)
    size_t waiting = 0;      ///< finished, waiting in the reorder buffer for an earlier seq
    size_t held_by_app = 0;  ///< delivered Results the application has not dropped yet
};

class PoolWarner {
public:
    /// extra_result_buffers: the network's spare blocks, shown in the hint (-1 = don't show).
    explicit PoolWarner(std::string network, std::ostream& os = std::cerr,
                        std::chrono::milliseconds min_interval = std::chrono::milliseconds(1000),
                        int extra_result_buffers = -1);

    /// Thread-safe. `occ` all zero = breakdown unknown (generic advice).
    void on_level(bool low, size_t free_blocks, size_t total_blocks, const PoolOccupancy& occ = PoolOccupancy());
    uint64_t low_events() const;  ///< Low episodes seen

private:
    mutable std::mutex m_;
    const std::string network_;
    std::ostream* const os_;
    const std::chrono::milliseconds interval_;
    const int extra_;
    bool printed_any_ = false;
    bool low_printed_ = false;
    std::chrono::steady_clock::time_point last_print_;
    std::chrono::steady_clock::time_point low_since_;
    uint64_t suppressed_ = 0;
    uint64_t low_events_ = 0;
};

}  // namespace dxinfer