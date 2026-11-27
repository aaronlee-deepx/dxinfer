// Optional performance measurement.
//
// The inference code calls three StatsRecorder hooks per job (feed, complete,
// deliver). When statistics are disabled every hook returns immediately:
// no clock is read and nothing is stored. Memory use is bounded (histograms
// with fixed buckets), so measurement can stay on for long runs.
#pragma once

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

namespace dxinfer {

using Clock = std::chrono::steady_clock;

/// Timestamps of one job. Only filled when statistics are enabled.
struct JobTimes {
    Clock::time_point fed;        ///< feed() accepted the input
    Clock::time_point completed;  ///< dx-runtime finished the job
};

/// Streaming latency statistics, values in microseconds.
/// count/mean/stddev/min/max are exact; percentiles come from log-scale
/// buckets (16 per power of two, about 4% resolution).
class LatencyHistogram {
public:
    LatencyHistogram();
    void add(double us);
    void reset();

    uint64_t count() const noexcept { return n_; }
    double mean() const noexcept { return n_ ? mean_ : 0.0; }
    double stddev() const noexcept;  ///< population standard deviation
    double min() const noexcept { return n_ ? min_ : 0.0; }
    double max() const noexcept { return n_ ? max_ : 0.0; }
    double percentile(double p) const;  ///< p in [0, 100]

private:
    static constexpr int kPerOctave = 16;
    static constexpr int kOctaves = 40;  // 1 us .. 2^40 us
    static constexpr int kBuckets = kPerOctave * kOctaves + 1;
    static int bucket_of(double us);
    static double bucket_value(int b);

    std::vector<uint64_t> buckets_;
    uint64_t n_ = 0;
    double mean_ = 0.0;
    double m2_ = 0.0;
    double min_ = 0.0;
    double max_ = 0.0;
};

struct LatencySummary {
    uint64_t count = 0;
    double mean = 0, stddev = 0, min = 0, max = 0, p50 = 0, p90 = 0, p99 = 0;  // microseconds
    static LatencySummary from(const LatencyHistogram& h);
};

/// Snapshot of one network's statistics.
struct NetworkStats {
    std::string network;
    bool enabled = false;

    uint64_t fed = 0, completed = 0, delivered = 0, failed = 0;
    int peak_in_flight = 0;      ///< highest number of jobs on the device at once
    int max_in_flight = 0;       ///< the limit (dxrt buffers x devices)
    double wall_us = 0;          ///< first feed -> last completion
    double fps = 0;              ///< completed / wall

    LatencySummary device;        ///< feed -> dx-runtime completion
    LatencySummary reorder_wait;  ///< completion -> released in FIFO order
    LatencySummary end_to_end;    ///< feed -> handed to you

    // dx-runtime engine counters (means in us, since the engine was created).
    // These are the numbers dxrun prints, for a direct comparison.
    bool has_engine_counters = false;
    double npu_us = 0, queue_wait_us = 0, dxrt_e2e_us = 0, core_latency_us = 0;
    int core_samples = 0;
};

void print_stats(std::ostream& os, const NetworkStats& s);

/// Collects the statistics of one network. Thread-safe.
class StatsRecorder {
public:
    explicit StatsRecorder(bool enabled) : enabled_(enabled) {}
    bool enabled() const noexcept { return enabled_; }

    void on_feed(JobTimes& t, int in_flight);  ///< sets t.fed
    void on_complete(JobTimes& t, bool ok);    ///< sets t.completed
    void on_deliver(const JobTimes& t);
    void reset();                              ///< call while idle
    NetworkStats snapshot() const;             ///< app-side fields

private:
    const bool enabled_;
    mutable std::mutex m_;
    uint64_t fed_ = 0, completed_ = 0, delivered_ = 0, failed_ = 0;
    int peak_ = 0;
    bool started_ = false;
    Clock::time_point first_feed_;
    Clock::time_point last_complete_;
    LatencyHistogram device_, reorder_, e2e_;
};

}  // namespace dxinfer