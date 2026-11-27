// Host-only tests for dxinfer::ReorderBuffer (no NPU, no dx-runtime).
#include "dxinfer/reorder_buffer.hpp"
#include "test_common.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

using dxinfer::ReorderBuffer;

static void shuffled_completions_come_out_in_order()
{
    ReorderBuffer<int> rb;
    std::vector<uint64_t> seqs(1000);
    std::iota(seqs.begin(), seqs.end(), 0);
    std::shuffle(seqs.begin(), seqs.end(), std::mt19937(42));
    for (uint64_t s : seqs) rb.push(s, static_cast<int>(s) * 10);

    for (int i = 0; i < 1000; ++i) {
        int v = -1;
        CHECK(rb.try_pop(v));
        CHECK(v == i * 10);
    }
    int v = 0;
    CHECK(!rb.try_pop(v));
    CHECK(rb.next_seq() == 1000);
    CHECK(rb.size() == 0);
}

static void gap_holds_back_later_items()
{
    ReorderBuffer<int> rb;
    rb.push(1, 1);
    rb.push(2, 2);
    int v = -1;
    CHECK(!rb.try_pop(v));          // seq 0 still missing
    CHECK(rb.size() == 2);
    rb.push(0, 0);
    for (int i = 0; i < 3; ++i) { CHECK(rb.try_pop(v)); CHECK(v == i); }
}

static void move_only_items()
{
    ReorderBuffer<std::unique_ptr<int>> rb;
    rb.push(1, std::unique_ptr<int>(new int(11)));
    rb.push(0, std::unique_ptr<int>(new int(10)));
    std::unique_ptr<int> p;
    CHECK(rb.try_pop(p) && p && *p == 10);
    CHECK(rb.try_pop(p) && p && *p == 11);
}

static void multi_producer_stress()
{
    ReorderBuffer<uint64_t> rb;
    const uint64_t kItems = 20000;
    const int kProducers = 4;
    std::atomic<int> errors{0};

    std::thread consumer([&] {
        for (uint64_t i = 0; i < kItems; ++i) {
            uint64_t v = 0;
            if (!rb.pop(v)) { ++errors; return; }
            if (v != i) ++errors;
        }
    });
    std::vector<std::thread> producers;
    for (int k = 0; k < kProducers; ++k) {
        producers.emplace_back([&rb, k, kItems] {
            std::mt19937 rng(static_cast<unsigned>(k));
            for (uint64_t s = static_cast<uint64_t>(k); s < kItems; s += kProducers) {
                if (rng() % 64 == 0) std::this_thread::yield();
                rb.push(s, s);
            }
        });
    }
    for (auto& t : producers) t.join();
    consumer.join();
    CHECK(errors == 0);
    CHECK(rb.next_seq() == kItems);
}

static void close_wakes_blocked_consumer()
{
    ReorderBuffer<int> rb;
    std::atomic<bool> returned{false};
    bool popped = true;
    std::thread t([&] {
        int v = 0;
        popped = rb.pop(v);
        returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(!returned);
    rb.close();
    t.join();
    CHECK(!popped);
    CHECK(rb.closed());
}

static void closed_queue_still_drains_ready_items()
{
    ReorderBuffer<int> rb;
    rb.push(0, 7);
    rb.close();
    int v = 0;
    CHECK(rb.pop(v) && v == 7);
    CHECK(!rb.pop(v));
}

static void pop_for_times_out()
{
    ReorderBuffer<int> rb;
    int v = 0;
    const auto t0 = std::chrono::steady_clock::now();
    CHECK(!rb.pop_for(v, std::chrono::milliseconds(20)));
    CHECK(std::chrono::steady_clock::now() - t0 >= std::chrono::milliseconds(15));
    rb.push(0, 3);
    CHECK(rb.pop_for(v, std::chrono::milliseconds(20)) && v == 3);
}

static void duplicate_or_old_seq_throws()
{
    ReorderBuffer<int> rb;
    rb.push(0, 0);
    int v = 0;
    CHECK(rb.try_pop(v));
    CHECK_THROWS(rb.push(0, 0), std::logic_error);   // already released
    rb.push(5, 5);
    CHECK_THROWS(rb.push(5, 5), std::logic_error);   // pending twice
}

int main()
{
    RUN(shuffled_completions_come_out_in_order);
    RUN(gap_holds_back_later_items);
    RUN(move_only_items);
    RUN(multi_producer_stress);
    RUN(close_wakes_blocked_consumer);
    RUN(closed_queue_still_drains_ready_items);
    RUN(pop_for_times_out);
    RUN(duplicate_or_old_seq_throws);
    return report("test_reorder_buffer");
}