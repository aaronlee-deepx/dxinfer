// Minimal test helpers (no external framework): CHECK records a failure and
// continues, RUN prints PASS/FAIL per test, report() gives the exit code.
#pragma once
#include <cstdio>

namespace testing_detail {
inline int& failures() { static int f = 0; return f; }
}

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "  CHECK failed %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                                 \
            ++testing_detail::failures();                                        \
        }                                                                        \
    } while (0)

#define CHECK_THROWS(expr, exception_type)                                       \
    do {                                                                         \
        bool thrown_ = false;                                                    \
        try { expr; } catch (const exception_type&) { thrown_ = true; }          \
        if (!thrown_) {                                                          \
            std::fprintf(stderr, "  CHECK_THROWS failed %s:%d: %s\n", __FILE__,    \
                         __LINE__, #expr);                                       \
            ++testing_detail::failures();                                        \
        }                                                                        \
    } while (0)

#define RUN(test)                                                                \
    do {                                                                         \
        const int before_ = testing_detail::failures();                          \
        test();                                                                  \
        std::printf("[%s] %s\n",                                                 \
                    testing_detail::failures() == before_ ? "PASS" : "FAIL", #test); \
    } while (0)

inline int report(const char* suite)
{
    const int f = testing_detail::failures();
    std::printf("%s: %s (%d failed check%s)\n", suite, f ? "FAIL" : "PASS", f, f == 1 ? "" : "s");
    return f ? 1 : 0;
}