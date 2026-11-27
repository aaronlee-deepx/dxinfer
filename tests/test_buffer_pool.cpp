// Host-only tests for dxinfer::BufferPool / PooledBuffer (no NPU, no dx-runtime).
#include "dxinfer/buffer_pool.hpp"
#include "test_common.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

using dxinfer::BufferPool;
using dxinfer::PooledBuffer;

static void exhausts_and_refills()
{
    auto pool = BufferPool::create(1000, 3);
    CHECK(pool->capacity() == 3);
    CHECK(pool->block_bytes() == 1000);
    std::vector<PooledBuffer> held;
    for (int i = 0; i < 3; ++i) {
        held.push_back(pool->acquire());
        CHECK(held.back());
        CHECK(held.back().size() == 1000);
    }
    CHECK(pool->available() == 0);
    CHECK(!pool->try_acquire());
    held.pop_back();                       // RAII: block goes back
    CHECK(pool->available() == 1);
    PooledBuffer b = pool->try_acquire();
    CHECK(b);
}

static void blocks_are_distinct_aligned_and_writable()
{
    auto pool = BufferPool::create(100, 4);
    std::vector<PooledBuffer> blocks;
    for (int i = 0; i < 4; ++i) blocks.push_back(pool->acquire());
    for (size_t i = 0; i < blocks.size(); ++i) {
        CHECK(reinterpret_cast<uintptr_t>(blocks[i].data()) % BufferPool::kAlignment == 0);
        std::memset(blocks[i].data(), static_cast<int>(i + 1), blocks[i].size());
        for (size_t j = 0; j < i; ++j) CHECK(blocks[i].data() != blocks[j].data());
    }
    for (size_t i = 0; i < blocks.size(); ++i)
        for (size_t k = 0; k < blocks[i].size(); ++k)
            if (blocks[i].data()[k] != static_cast<uint8_t>(i + 1)) { CHECK(false); break; }
}

static void blocked_acquire_wakes_on_release()
{
    auto pool = BufferPool::create(16, 1);
    PooledBuffer a = pool->acquire();
    std::atomic<bool> got{false};
    std::thread t([&] {
        PooledBuffer b = pool->acquire();
        got = static_cast<bool>(b);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!got);
    a.reset();
    t.join();
    CHECK(got);
}

static void shutdown_wakes_waiters_with_empty_handle()
{
    auto pool = BufferPool::create(16, 1);
    PooledBuffer a = pool->acquire();
    std::atomic<bool> done{false};
    bool empty = false;
    std::thread t([&] {
        PooledBuffer b = pool->acquire();
        empty = !b;
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(!done);
    pool->shutdown();
    t.join();
    CHECK(empty);
    CHECK(!pool->try_acquire());
}

static void handle_outlives_pool_owner()
{
    PooledBuffer b;
    {
        auto pool = BufferPool::create(32, 1);
        b = pool->acquire();
    }                                       // last user-side shared_ptr gone
    CHECK(b);
    b.data()[0] = 1;                        // block still valid
    b.reset();
    CHECK(!b);
}

static void move_semantics()
{
    auto pool = BufferPool::create(8, 1);
    PooledBuffer a = pool->acquire();
    uint8_t* p = a.data();
    PooledBuffer b(std::move(a));
    CHECK(!a);
    CHECK(b.data() == p);
    a = std::move(b);
    CHECK(a.data() == p);
    CHECK(pool->available() == 0);
}

int main()
{
    RUN(exhausts_and_refills);
    RUN(blocks_are_distinct_aligned_and_writable);
    RUN(blocked_acquire_wakes_on_release);
    RUN(shutdown_wakes_waiters_with_empty_handle);
    RUN(handle_outlives_pool_owner);
    RUN(move_semantics);
    return report("test_buffer_pool");
}