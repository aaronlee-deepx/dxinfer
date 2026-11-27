// Host-only tests for dxinfer::Result::copy_out() (no NPU, no dx-runtime).
#include "dxinfer/result.hpp"
#include "test_common.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace dxinfer;

// Build a pool-backed Result the way the library does: two tensors inside one block.
static Result make_pooled(const std::shared_ptr<BufferPool>& pool)
{
    PooledBuffer block = pool->acquire();
    for (size_t i = 0; i < block.size(); ++i) block.data()[i] = static_cast<uint8_t>(i);
    std::vector<TensorView> views(2);
    views[0].name = "boxes";
    views[0].shape = {1, 8};
    views[0].data = block.data();
    views[0].bytes = 8;
    views[1].name = "scores";
    views[1].shape = {1, 4};
    views[1].data = block.data() + 16;
    views[1].bytes = 4;
    JobTimes t;
    t.fed = Clock::now();
    return detail::ResultAccess::make(std::make_shared<const std::string>("det"), 7,
                                      reinterpret_cast<void*>(static_cast<uintptr_t>(42)), Status(),
                                      std::move(block), std::move(views), 64, t);
}

static void copy_has_same_metadata_and_bytes()
{
    auto pool = BufferPool::create(64, 2);
    Result r = make_pooled(pool);
    Result c = r.copy_out();
    CHECK(r.uses_pool());
    CHECK(!c.uses_pool());
    CHECK(c.network() == "det" && c.seq() == 7 && c.user_arg() == r.user_arg() && c.ok());
    CHECK(c.bytes() == 64 && c.tensor_count() == 2);
    CHECK(c.times().fed == r.times().fed);
    CHECK(c.tensor("scores").shape == r.tensor("scores").shape);
    CHECK(std::memcmp(c.data(), r.data(), 64) == 0);
    // views are rebased into the copy, at the same offsets
    CHECK(c.tensor(0).data == c.data());
    CHECK(c.tensor(1).data == c.data() + 16);
    CHECK(c.tensor(1).data != r.tensor(1).data);
    CHECK(c.tensor(1).as<uint8_t>()[0] == 16);
}

static void copy_holds_no_block_and_outlives_original()
{
    auto pool = BufferPool::create(64, 1);
    Result c;
    {
        Result r = make_pooled(pool);
        CHECK(pool->available() == 0);
        c = r.copy_out();
    }                                       // original dropped -> block back in the pool
    CHECK(pool->available() == 1);
    CHECK(c.tensor(0).as<uint8_t>()[3] == 3);
    Result moved = std::move(c);            // views stay valid after a move
    CHECK(moved.tensor(1).as<uint8_t>()[0] == 16);
    Result again = moved.copy_out();        // copy of a copy works too
    CHECK(!again.uses_pool() && again.tensor(1).as<uint8_t>()[2] == 18);
}

static void copy_of_failed_result()
{
    Result failed = detail::ResultAccess::make(std::make_shared<const std::string>("cls"), 3, nullptr,
                                               Status(ErrorCode::InferenceFailed, "boom"), PooledBuffer(),
                                               std::vector<TensorView>(), 0, JobTimes());
    Result c = failed.copy_out();
    CHECK(!c.ok() && c.status().message() == "boom" && c.seq() == 3 && c.tensor_count() == 0);
}

int main()
{
    RUN(copy_has_same_metadata_and_bytes);
    RUN(copy_holds_no_block_and_outlives_original);
    RUN(copy_of_failed_result);
    return report("test_result");
}