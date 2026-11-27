// demo_callback - CallbackInference: one callback per network checks FIFO order,
// main thread feeds every network round-robin, then wait_idle().
#include "demo_common.hpp"

#include <mutex>

using namespace dxinfer;

int main(int argc, char** argv)
{
    demo::Args a;
    const int rc = demo::parse(argc, argv, a);
    if (rc) return rc == 2 ? 0 : 1;

    // Declared before `inf` so the callbacks' state outlives the inference object.
    std::vector<demo::Check> checks(a.nets.size());
    std::mutex print_m;

    try {
        CallbackInference inf(a.nets, a.cfg);
        inf.print_info();

        const std::vector<std::string> names = inf.network_names();
        std::vector<std::vector<uint8_t>> inputs;
        for (size_t k = 0; k < names.size(); ++k) {
            inputs.emplace_back(inf.input_size(names[k]), 0);  // fake data
            demo::Check* c = &checks[k];
            // Runs on this network's delivery thread, one result at a time, in FIFO order.
            inf.set_callback(names[k], [c, &print_m](Result& r) {
                if (r.seq() != c->received || r.user_arg() != demo::tag(c->received)) ++c->order_errors;
                if (!r.ok()) ++c->failed;
                if (c->received == 0) {
                    std::lock_guard<std::mutex> lk(print_m);
                    demo::print_first_result(r);
                    if (!r.ok()) std::cout << "     error: " << r.status().message() << "\n";
                }
                ++c->received;
            });
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < a.loops; ++i)
            for (size_t k = 0; k < names.size(); ++k)
                inf.feed(names[k], inputs[k].data(), inputs[k].size(), demo::tag(static_cast<uint64_t>(i)));
        inf.wait_idle();
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        bool ok = demo::print_summary(names, checks, a.loops, wall);
        for (const auto& n : names)
            if (inf.callback_errors(n)) { ok = false; std::cout << "  " << n << ": callback errors " << inf.callback_errors(n) << "\n"; }
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