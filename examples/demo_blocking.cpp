// demo_blocking - BlockingInference: main thread feeds every network round-robin,
// one consumer thread per network pulls results with get_for() and checks FIFO order.
#include "demo_common.hpp"

#include <atomic>
#include <mutex>
#include <thread>

using namespace dxinfer;

int main(int argc, char** argv)
{
    demo::Args a;
    const int rc = demo::parse(argc, argv, a);
    if (rc) return rc == 2 ? 0 : 1;

    try {
        BlockingInference inf(a.nets, a.cfg);
        inf.print_info();

        const std::vector<std::string> names = inf.network_names();
        std::vector<std::vector<uint8_t>> inputs;
        for (const auto& n : names) inputs.emplace_back(inf.input_size(n), 0);  // fake data

        std::vector<demo::Check> checks(names.size());
        std::atomic<bool> stop{false};
        std::mutex print_m;
        std::vector<std::thread> consumers;
        const auto t0 = std::chrono::steady_clock::now();

        for (size_t k = 0; k < names.size(); ++k) {
            consumers.emplace_back([&, k] {
                demo::Check& c = checks[k];
                while (c.received < static_cast<uint64_t>(a.loops) && !stop) {
                    Result r;
                    if (!inf.get_for(names[k], std::chrono::milliseconds(100), r)) continue;
                    if (r.seq() != c.received || r.user_arg() != demo::tag(c.received)) ++c.order_errors;
                    if (!r.ok()) ++c.failed;
                    if (c.received == 0) {
                        std::lock_guard<std::mutex> lk(print_m);
                        demo::print_first_result(r);
                        if (!r.ok()) std::cout << "     error: " << r.status().message() << "\n";
                    }
                    ++c.received;
                }
            });
        }

        try {
            for (int i = 0; i < a.loops; ++i)
                for (size_t k = 0; k < names.size(); ++k)
                    inf.feed(names[k], inputs[k].data(), inputs[k].size(), demo::tag(static_cast<uint64_t>(i)));
        } catch (...) {
            stop = true;
            for (auto& t : consumers) t.join();
            throw;
        }
        for (auto& t : consumers) t.join();
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        const bool ok = demo::print_summary(names, checks, a.loops, wall);
        if (inf.stats_enabled()) inf.print_stats();
        return ok ? 0 : 1;
    } catch (const InferenceError& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}