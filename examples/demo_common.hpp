// Shared helpers for the two demos: command line, fake input, summary.
#pragma once

#include "dxinfer/dxinfer.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace demo {

struct Args {
    std::vector<dxinfer::NetworkSpec> nets;
    dxinfer::InferenceConfig cfg;
    int loops = 100;  // jobs per network
};

inline void usage(const char* prog)
{
    std::printf(
        "Usage: %s -n NAME=MODEL.dxnn [-n NAME2=MODEL2.dxnn ...] [options]\n"
        "  -n, --network NAME=PATH  network to load (repeat for several; names must be unique)\n"
        "  -e, --exec sync|async    dx-runtime mode underneath (default async)\n"
        "  -l, --loops N            jobs per network (default 100)\n"
        "  -s, --stats              enable performance statistics\n"
        "  -b, --buffer-count N     dxrt buffers per device for every network (1-100, default 6)\n"
        "      --extra-buffers N    result blocks beyond max-in-flight (default = max-in-flight)\n"
        "  -o, --ort auto|on|off    CPU (ONNX Runtime) tasks for every network (default auto: from the model graph)\n"
        "  -h, --help\n"
        "Input is zero-filled (pure inference, no pre/post-processing).\n",
        prog);
}

inline bool to_int(const std::string& s, int lo, int hi, int& out)
{
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || v < lo || v > hi) return false;
    out = static_cast<int>(v);
    return true;
}

/// 0 = run, 1 = error, 2 = help shown
inline int parse(int argc, char** argv, Args& a)
{
    int buffer_count = 0;
    dxinfer::OrtMode ort = dxinfer::OrtMode::Auto;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto value = [&](std::string& v) -> bool {
            if (i + 1 >= argc) { std::fprintf(stderr, "[error] %s needs a value\n", k.c_str()); return false; }
            v = argv[++i];
            return true;
        };
        std::string v;
        if (k == "-h" || k == "--help") { usage(argv[0]); return 2; }
        else if (k == "-s" || k == "--stats") a.cfg.enable_stats = true;
        else if (k == "-n" || k == "--network") {
            if (!value(v)) return 1;
            const size_t eq = v.find('=');
            if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) {
                std::fprintf(stderr, "[error] -n expects NAME=PATH, got '%s'\n", v.c_str());
                return 1;
            }
            a.nets.emplace_back(v.substr(0, eq), v.substr(eq + 1));
        } else if (k == "-e" || k == "--exec") {
            if (!value(v)) return 1;
            if (v == "sync") a.cfg.exec = dxinfer::ExecMode::Sync;
            else if (v == "async") a.cfg.exec = dxinfer::ExecMode::Async;
            else { std::fprintf(stderr, "[error] --exec must be sync or async\n"); return 1; }
        } else if (k == "-l" || k == "--loops") {
            if (!value(v) || !to_int(v, 1, 100000000, a.loops)) { std::fprintf(stderr, "[error] bad --loops\n"); return 1; }
        } else if (k == "-b" || k == "--buffer-count") {
            if (!value(v) || !to_int(v, 1, 100, buffer_count)) { std::fprintf(stderr, "[error] --buffer-count must be 1-100\n"); return 1; }
        } else if (k == "-o" || k == "--ort") {
            if (!value(v)) return 1;
            if (v == "auto") ort = dxinfer::OrtMode::Auto;
            else if (v == "on") ort = dxinfer::OrtMode::On;
            else if (v == "off") ort = dxinfer::OrtMode::Off;
            else { std::fprintf(stderr, "[error] --ort must be auto, on or off\n"); return 1; }
        } else if (k == "--extra-buffers") {
            if (!value(v) || !to_int(v, 0, 10000, a.cfg.extra_result_buffers)) { std::fprintf(stderr, "[error] bad --extra-buffers\n"); return 1; }
        } else {
            std::fprintf(stderr, "[error] unknown option '%s' (see --help)\n", k.c_str());
            return 1;
        }
    }
    if (a.nets.empty()) { std::fprintf(stderr, "[error] at least one -n NAME=MODEL.dxnn is required\n"); return 1; }
    for (auto& n : a.nets) {
        n.options.buffer_count = buffer_count;
        n.options.ort = ort;
    }
    return 0;
}

/// Tag carried through user_arg so the demo can check it comes back unchanged.
inline void* tag(uint64_t i) { return reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1)); }

inline void print_first_result(const dxinfer::Result& r)
{
    std::cout << "first result of '" << r.network() << "': status " << dxinfer::to_string(r.status().code())
              << ", " << r.tensor_count() << " tensor(s)\n";
    for (const auto& t : r.tensors()) {
        std::cout << "     " << t.name << "  [";
        for (size_t i = 0; i < t.shape.size(); ++i) std::cout << (i ? "," : "") << t.shape[i];
        std::cout << "]  " << t.bytes << " B\n";
    }
}

struct Check {
    uint64_t received = 0;
    uint64_t order_errors = 0;  // seq or user_arg not as fed
    uint64_t failed = 0;        // Result::ok() == false
};

inline bool print_summary(const std::vector<std::string>& names, const std::vector<Check>& checks, int loops,
                          double wall_s)
{
    bool all_ok = true;
    uint64_t total = 0;
    std::cout << "\n=== summary ===\n";
    for (size_t i = 0; i < names.size(); ++i) {
        const Check& c = checks[i];
        const bool ok = c.received == static_cast<uint64_t>(loops) && c.order_errors == 0 && c.failed == 0;
        all_ok = all_ok && ok;
        total += c.received;
        std::cout << "  " << names[i] << ": received " << c.received << "/" << loops << ", FIFO order "
                  << (c.order_errors ? "BROKEN" : "OK") << ", failed " << c.failed << "  -> "
                  << (ok ? "OK" : "FAIL") << "\n";
    }
    std::cout << "  total " << total << " results in " << wall_s * 1000.0 << " ms -> "
              << (wall_s > 0 ? static_cast<double>(total) / wall_s : 0.0) << " results/s (all networks)\n";
    return all_ok;
}

}  // namespace demo