// Host-only tests for dxinfer::LatencyHistogram / StatsRecorder (no NPU, no dx-runtime).
#include "dxinfer/stats.hpp"
#include "test_common.hpp"

#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <thread>

using namespace dxinfer;

static bool near(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::fabs(b); }

static void histogram_exact_moments()
{
    LatencyHistogram h;
    for (double v : {10.0, 20.0, 30.0, 40.0}) h.add(v);
    CHECK(h.count() == 4);
    CHECK(near(h.mean(), 25.0, 1e-12));
    CHECK(h.min() == 10.0);
    CHECK(h.max() == 40.0);
    CHECK(near(h.stddev(), std::sqrt(125.0), 1e-9));   // population stddev
}

static void histogram_percentiles_within_resolution()
{
    LatencyHistogram h;
    for (int i = 1; i <= 10000; ++i) h.add(i);
    CHECK(near(h.percentile(50), 5000, 0.05));
    CHECK(near(h.percentile(90), 9000, 0.05));
    CHECK(near(h.percentile(99), 9900, 0.05));
    CHECK(h.percentile(0) >= h.min());
    CHECK(h.percentile(100) <= h.max());
}

static void histogram_reset_and_empty()
{
    LatencyHistogram h;
    CHECK(h.count() == 0 && h.mean() == 0 && h.percentile(50) == 0 && h.stddev() == 0);
    h.add(5);
    h.reset();
    CHECK(h.count() == 0 && h.max() == 0);
}

static void histogram_extremes()
{
    LatencyHistogram h;
    h.add(0.2);
    h.add(1e12);
    CHECK(h.count() == 2);
    CHECK(h.max() == 1e12);
    CHECK(h.percentile(100) <= h.max());
    CHECK(h.percentile(1) >= h.min());
}

static void disabled_recorder_records_nothing()
{
    StatsRecorder r(false);
    JobTimes t;
    r.on_feed(t, 3);
    r.on_complete(t, true);
    r.on_deliver(t);
    CHECK(t.fed == Clock::time_point());      // no clock read at all
    const NetworkStats s = r.snapshot();
    CHECK(!s.enabled);
    CHECK(s.fed == 0 && s.completed == 0 && s.delivered == 0);
    std::ostringstream os;
    print_stats(os, s);
    CHECK(os.str().find("disabled") != std::string::npos);
}

static void enabled_recorder_measures()
{
    StatsRecorder r(true);
    JobTimes a, b;
    r.on_feed(a, 1);
    r.on_feed(b, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    r.on_complete(b, true);            // out of order on purpose
    r.on_complete(a, false);
    r.on_deliver(a);
    r.on_deliver(b);
    NetworkStats s = r.snapshot();
    CHECK(s.enabled);
    CHECK(s.fed == 2 && s.completed == 2 && s.delivered == 2 && s.failed == 1);
    CHECK(s.peak_in_flight == 2);
    CHECK(s.device.count == 2 && s.device.min >= 1500.0);
    CHECK(s.end_to_end.count == 2 && s.end_to_end.min >= s.device.min * 0.99);
    CHECK(s.wall_us > 0 && s.fps > 0);
    std::ostringstream os;
    print_stats(os, s);
    CHECK(os.str().find("FPS") != std::string::npos);

    r.reset();
    s = r.snapshot();
    CHECK(s.fed == 0 && s.completed == 0 && s.device.count == 0 && s.fps == 0);
}

int main()
{
    RUN(histogram_exact_moments);
    RUN(histogram_percentiles_within_resolution);
    RUN(histogram_reset_and_empty);
    RUN(histogram_extremes);
    RUN(disabled_recorder_records_nothing);
    RUN(enabled_recorder_measures);
    return report("test_stats");
}