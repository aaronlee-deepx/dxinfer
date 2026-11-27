// Host-only tests for the low result-buffer warning: BufferPool level watch + PoolWarner.
#include "dxinfer/buffer_pool.hpp"
#include "dxinfer/pool_warning.hpp"
#include "test_common.hpp"

#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace dxinfer;

struct Event { bool low; size_t free_blocks; };

static std::shared_ptr<const BufferPool::LevelFn> recorder(std::vector<Event>& ev)
{
    return std::make_shared<const BufferPool::LevelFn>(
        [&ev](bool low, size_t free_blocks, size_t) { ev.push_back(Event{low, free_blocks}); });
}

static void watch_is_edge_triggered_with_hysteresis()
{
    auto pool = BufferPool::create(8, 5);
    std::vector<Event> ev;
    pool->set_level_watch(1, 3, recorder(ev));

    std::vector<PooledBuffer> held;
    for (int i = 0; i < 3; ++i) held.push_back(pool->acquire());   // free 2
    CHECK(ev.empty());
    held.push_back(pool->acquire());                                // free 1 -> Low
    CHECK(ev.size() == 1 && ev[0].low && ev[0].free_blocks == 1);
    held.push_back(pool->acquire());                                // free 0 -> still low, no event
    CHECK(ev.size() == 1);
    held.pop_back();                                                // free 1
    held.pop_back();                                                // free 2 (below recover mark)
    CHECK(ev.size() == 1);
    held.pop_back();                                                // free 3 -> Recovered
    CHECK(ev.size() == 2 && !ev[1].low && ev[1].free_blocks == 3);
    held.push_back(pool->acquire());                                // free 2
    held.push_back(pool->acquire());                                // free 1 -> Low again
    CHECK(ev.size() == 3 && ev[2].low);
}

static void cleared_watch_reports_nothing()
{
    auto pool = BufferPool::create(8, 2);
    std::vector<Event> ev;
    pool->set_level_watch(1, 2, recorder(ev));
    pool->set_level_watch(0, 0, nullptr);
    PooledBuffer a = pool->acquire(), b = pool->acquire();
    a.reset();
    b.reset();
    CHECK(ev.empty());
}

static void warner_prints_low_then_recovered()
{
    std::ostringstream os;
    PoolWarner w("det", os, std::chrono::milliseconds(1000));
    w.on_level(true, 1, 12);
    const std::string low = os.str();
    CHECK(low.find("WARNING") != std::string::npos);
    CHECK(low.find("det:") != std::string::npos);
    CHECK(low.find("1 of 12") != std::string::npos);
    CHECK(low.find("copy_out") != std::string::npos);
    w.on_level(false, 3, 12);
    CHECK(os.str().find("result blocks recovered (3 of 12 free)") != std::string::npos);
    CHECK(w.low_events() == 1);
}

static void warner_is_rate_limited_and_counts_suppressed()
{
    std::ostringstream os;
    PoolWarner w("det", os, std::chrono::milliseconds(80));
    w.on_level(true, 1, 12);     // printed
    w.on_level(false, 3, 12);    // printed (its Low was printed)
    const size_t len = os.str().size();
    w.on_level(true, 0, 12);     // within 80 ms -> suppressed
    w.on_level(false, 4, 12);    // its Low was not printed -> silent
    CHECK(os.str().size() == len);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    w.on_level(true, 1, 12);     // printed, mentions the suppressed one
    CHECK(os.str().find("1 similar warning suppressed") != std::string::npos);
    CHECK(w.low_events() == 3);
}

static size_t lines(const std::string& s)
{
    size_t n = 0;
    for (char c : s) n += (c == '\n');
    return n;
}

static void warner_names_fifo_wait_as_cause()
{
    std::ostringstream os;
    PoolWarner w("det", os, std::chrono::milliseconds(1000), /*extra*/ 6);
    w.on_level(true, 1, 12, PoolOccupancy{6, 5, 0});
    const std::string m = os.str();
    CHECK(m.find("result blocks low (1 of 12 free)") != std::string::npos);
    CHECK(m.find("in flight 6 | waiting for FIFO order 5 | held by your app 0") != std::string::npos);
    CHECK(m.find("earlier job is slow") != std::string::npos);
    CHECK(m.find("extra_result_buffers (now 6)") != std::string::npos);
    CHECK(m.find("your app is still holding") == std::string::npos);
    CHECK(m.find("The application is holding") == std::string::npos);  // old, misleading text
    CHECK(lines(m) <= 4);
}

static void warner_names_app_as_cause()
{
    std::ostringstream os;
    PoolWarner w("det", os, std::chrono::milliseconds(1000), 6);
    w.on_level(true, 1, 12, PoolOccupancy{6, 0, 5});
    const std::string m = os.str();
    CHECK(m.find("in flight 6 | waiting for FIFO order 0 | held by your app 5") != std::string::npos);
    CHECK(m.find("your app is still holding 5 Results") != std::string::npos);
    CHECK(m.find("copy_out") != std::string::npos);
    CHECK(m.find("earlier job is slow") == std::string::npos);
    CHECK(lines(m) <= 4);
}

int main()
{
    RUN(watch_is_edge_triggered_with_hysteresis);
    RUN(cleared_watch_reports_nothing);
    RUN(warner_prints_low_then_recovered);
    RUN(warner_is_rate_limited_and_counts_suppressed);
    RUN(warner_names_fifo_wait_as_cause);
    RUN(warner_names_app_as_cause);
    return report("test_pool_warning");
}