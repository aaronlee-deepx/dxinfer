#include "dxinfer/stats.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <ostream>

namespace dxinfer {

constexpr int LatencyHistogram::kPerOctave;
constexpr int LatencyHistogram::kOctaves;
constexpr int LatencyHistogram::kBuckets;

// ---------------------------------------------------------------- histogram

LatencyHistogram::LatencyHistogram() : buckets_(kBuckets, 0) {}

// bucket 0: < 1 us; bucket b >= 1 covers [2^((b-1)/k), 2^(b/k)) us
int LatencyHistogram::bucket_of(double us)
{
    if (!(us >= 1.0)) return 0;
    const int b = 1 + static_cast<int>(std::floor(std::log2(us) * kPerOctave));
    return std::min(b, kBuckets - 1);
}

double LatencyHistogram::bucket_value(int b)
{
    if (b == 0) return 0.5;
    return std::exp2((b - 0.5) / kPerOctave);  // geometric centre of the bucket
}

void LatencyHistogram::add(double us)
{
    if (us < 0) us = 0;
    ++n_;
    if (n_ == 1) {
        min_ = max_ = us;
    } else {
        min_ = std::min(min_, us);
        max_ = std::max(max_, us);
    }
    const double delta = us - mean_;  // Welford
    mean_ += delta / static_cast<double>(n_);
    m2_ += delta * (us - mean_);
    ++buckets_[static_cast<size_t>(bucket_of(us))];
}

void LatencyHistogram::reset()
{
    std::fill(buckets_.begin(), buckets_.end(), 0);
    n_ = 0;
    mean_ = m2_ = min_ = max_ = 0.0;
}

double LatencyHistogram::stddev() const noexcept
{
    return n_ > 1 ? std::sqrt(m2_ / static_cast<double>(n_)) : 0.0;
}

double LatencyHistogram::percentile(double p) const
{
    if (n_ == 0) return 0.0;
    p = std::max(0.0, std::min(100.0, p));
    uint64_t rank = static_cast<uint64_t>(std::ceil(p / 100.0 * static_cast<double>(n_)));
    rank = std::max<uint64_t>(rank, 1);
    uint64_t cum = 0;
    for (int b = 0; b < kBuckets; ++b) {
        cum += buckets_[static_cast<size_t>(b)];
        if (cum >= rank) return std::max(min_, std::min(max_, bucket_value(b)));
    }
    return max_;
}

LatencySummary LatencySummary::from(const LatencyHistogram& h)
{
    LatencySummary s;
    s.count = h.count();
    s.mean = h.mean();
    s.stddev = h.stddev();
    s.min = h.min();
    s.max = h.max();
    s.p50 = h.percentile(50);
    s.p90 = h.percentile(90);
    s.p99 = h.percentile(99);
    return s;
}

// ---------------------------------------------------------------- recorder

static double micros(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::micro>(b - a).count();
}

void StatsRecorder::on_feed(JobTimes& t, int in_flight)
{
    if (!enabled_) return;
    t.fed = Clock::now();
    std::lock_guard<std::mutex> lk(m_);
    if (!started_) {
        started_ = true;
        first_feed_ = t.fed;
    }
    ++fed_;
    peak_ = std::max(peak_, in_flight);
}

void StatsRecorder::on_complete(JobTimes& t, bool ok)
{
    if (!enabled_) return;
    t.completed = Clock::now();
    const double us = micros(t.fed, t.completed);
    std::lock_guard<std::mutex> lk(m_);
    ++completed_;
    if (!ok) ++failed_;
    if (t.completed > last_complete_) last_complete_ = t.completed;
    device_.add(us);
}

void StatsRecorder::on_deliver(const JobTimes& t)
{
    if (!enabled_) return;
    const Clock::time_point now = Clock::now();
    std::lock_guard<std::mutex> lk(m_);
    ++delivered_;
    reorder_.add(micros(t.completed, now));
    e2e_.add(micros(t.fed, now));
}

void StatsRecorder::reset()
{
    std::lock_guard<std::mutex> lk(m_);
    fed_ = completed_ = delivered_ = failed_ = 0;
    peak_ = 0;
    started_ = false;
    first_feed_ = last_complete_ = Clock::time_point();
    device_.reset();
    reorder_.reset();
    e2e_.reset();
}

NetworkStats StatsRecorder::snapshot() const
{
    NetworkStats s;
    s.enabled = enabled_;
    if (!enabled_) return s;
    std::lock_guard<std::mutex> lk(m_);
    s.fed = fed_;
    s.completed = completed_;
    s.delivered = delivered_;
    s.failed = failed_;
    s.peak_in_flight = peak_;
    s.wall_us = (completed_ > 0 && last_complete_ > first_feed_) ? micros(first_feed_, last_complete_) : 0.0;
    s.fps = s.wall_us > 0 ? static_cast<double>(completed_) * 1e6 / s.wall_us : 0.0;
    s.device = LatencySummary::from(device_);
    s.reorder_wait = LatencySummary::from(reorder_);
    s.end_to_end = LatencySummary::from(e2e_);
    return s;
}

// ---------------------------------------------------------------- printing

static void print_row(std::ostream& os, const char* label, const LatencySummary& l)
{
    os << "    " << std::left << std::setw(14) << label << std::right;
    for (double v : {l.mean, l.p50, l.p90, l.p99, l.min, l.max}) os << std::setw(9) << v / 1000.0;
    os << "\n";
}

void print_stats(std::ostream& os, const NetworkStats& s)
{
    const std::ios::fmtflags flags = os.flags();
    const std::streamsize prec = os.precision();
    os << "=== stats: " << (s.network.empty() ? "?" : s.network) << " ===\n";
    if (!s.enabled) {
        os << "  (disabled - construct with InferenceConfig::enable_stats = true)\n";
        return;
    }
    os << std::fixed << std::setprecision(3);
    os << "  jobs          : fed " << s.fed << ", completed " << s.completed << ", delivered " << s.delivered
       << ", failed " << s.failed << "\n";
    os << "  in flight     : peak " << s.peak_in_flight << " / max " << s.max_in_flight << "\n";
    os << "  FPS           : " << std::setprecision(1) << s.fps << "   (" << s.completed << " jobs in "
       << std::setprecision(3) << s.wall_us / 1000.0 << " ms, first feed -> last completion)\n";
    os << "  latency [ms]       mean      p50      p90      p99      min      max\n";
    print_row(os, "device", s.device);
    print_row(os, "reorder wait", s.reorder_wait);
    print_row(os, "end-to-end", s.end_to_end);
    if (s.has_engine_counters) {
        os << "  dx-runtime counters (mean since engine creation, same lines as dxrun):\n";
        os << "    NPU Processing Time Average : " << s.npu_us / 1000.0 << " ms\n";
        os << "    Queue Wait Time Average     : " << s.queue_wait_us / 1000.0 << " ms\n";
        os << "    E2E Response Time Average   : " << s.dxrt_e2e_us / 1000.0 << " ms\n";
        os << "    Latency Average             : " << s.core_latency_us / 1000.0 << " ms (" << s.core_samples
           << " samples)\n";
    }
    os.flags(flags);
    os.precision(prec);
}

}  // namespace dxinfer