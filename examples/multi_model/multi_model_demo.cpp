// multi_model_demo - dxinfer CallbackInference (async) with two networks at once:
//   "det" yolo26-n (object detection) and "cls" resnet18 (image classification).
// 1. load both networks (all-or-nothing, auto ORT)   2. preprocess every image up front
// 3. one callback per network: post-process, draw the overlay, save it (runs on that network's lane thread)
// 4. one feeder thread per network pushes all inputs back to back   5. wait_idle(), summary + results.txt
#include "model_factory.hpp"
#include "vision.hpp"

#include <dxinfer/dxinfer.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

using namespace dxinfer;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
    std::string det = "models/yolo26-n_640x640.dxnn";
    std::string cls = "models/resnet18_224x224.dxnn";
    std::string det_dir = "images/image_object_detection";
    std::string cls_dir = "images/image_classification";
    std::string out = "output";
    int count = 10;
    float score = 0.3f;
    OrtMode ort = OrtMode::Auto;
    bool stats = false;
};

void usage(const char* p)
{
    std::printf(
        "Usage: %s [options]\n"
        "  --det MODEL      object detection model   (default models/yolo26-n_640x640.dxnn)\n"
        "  --cls MODEL      classification model     (default models/resnet18_224x224.dxnn)\n"
        "  --det-dir DIR    detection images         (default images/image_object_detection)\n"
        "  --cls-dir DIR    classification images    (default images/image_classification)\n"
        "  -o, --out DIR    output directory         (default output; overlays in DIR/det and DIR/cls)\n"
        "  -n, --count N    images per network, at least 10 (default 10)\n"
        "  --score S        detection score threshold (default 0.3)\n"
        "  --ort auto|on|off  ONNX Runtime for both networks (default auto)\n"
        "  -s, --stats      print dxinfer statistics\n"
        "Get images with ./get_images.sh. Models come from the DEEPX model zoo.\n",
        p);
}

int parse(int argc, char** argv, Args& a)  // 0 run, 1 error, 2 help
{
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&](std::string& v) {
            if (i + 1 >= argc) { std::fprintf(stderr, "[error] %s needs a value\n", k.c_str()); return false; }
            v = argv[++i];
            return true;
        };
        std::string v;
        if (k == "-h" || k == "--help") { usage(argv[0]); return 2; }
        else if (k == "--det") { if (!val(a.det)) return 1; }
        else if (k == "--cls") { if (!val(a.cls)) return 1; }
        else if (k == "--det-dir") { if (!val(a.det_dir)) return 1; }
        else if (k == "--cls-dir") { if (!val(a.cls_dir)) return 1; }
        else if (k == "-o" || k == "--out") { if (!val(a.out)) return 1; }
        else if (k == "-s" || k == "--stats") a.stats = true;
        else if (k == "-n" || k == "--count") {
            if (!val(v)) return 1;
            a.count = std::atoi(v.c_str());
            if (a.count < 10) { std::fprintf(stderr, "[error] --count must be at least 10\n"); return 1; }
        } else if (k == "--score") {
            if (!val(v)) return 1;
            a.score = static_cast<float>(std::atof(v.c_str()));
            if (a.score <= 0.f || a.score >= 1.f) { std::fprintf(stderr, "[error] --score must be in (0, 1)\n"); return 1; }
        } else if (k == "--ort") {
            if (!val(v)) return 1;
            if (v == "auto") a.ort = OrtMode::Auto;
            else if (v == "on") a.ort = OrtMode::On;
            else if (v == "off") a.ort = OrtMode::Off;
            else { std::fprintf(stderr, "[error] --ort must be auto, on or off\n"); return 1; }
        } else { std::fprintf(stderr, "[error] unknown option '%s' (see --help)\n", k.c_str()); return 1; }
    }
    return 0;
}

bool file_exists(const std::string& p)
{
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

/// One image of one network: the original (for the overlay) and its preprocessed input.
struct Job {
    std::string path;
    cv::Mat original;
    mm::Prepared prepared;
};

/// What one network's callback records (only its own lane thread writes it).
struct Lane {
    std::vector<std::string> lines;  // per-image summary, in feed order
    uint64_t received = 0, order_errors = 0, failed = 0, saved = 0;
};

/// Input geometry from the loaded network: [1, H, W, 3] UINT8 (NHWC), as dx_app's models expect.
void input_hw(const NetworkInfo& ni, int* w, int* h)
{
    if (ni.inputs.size() != 1 || ni.inputs[0].shape.size() != 4 || ni.inputs[0].shape[3] != 3 ||
        ni.inputs[0].dtype_name != "UINT8")
        throw std::runtime_error("network '" + ni.name + "': expected one UINT8 input of shape [1,H,W,3]");
    *h = static_cast<int>(ni.inputs[0].shape[1]);
    *w = static_cast<int>(ni.inputs[0].shape[2]);
}

/// The first `count` images of `dir` (sorted), or throws if there are fewer. Checked before the device is touched.
std::vector<std::string> pick_images(const std::string& net, const std::string& dir, int count)
{
    std::vector<std::string> files = mm::list_images(dir);
    if (static_cast<int>(files.size()) < count)
        throw std::runtime_error(net + ": need at least " + std::to_string(count) + " images in " + dir + ", found " +
                                 std::to_string(files.size()) + " (run ./get_images.sh)");
    files.resize(static_cast<size_t>(count));
    return files;
}

std::vector<Job> prepare(const std::string& net, const std::vector<std::string>& files, const mm::IPreprocessor& pre,
                         uint64_t expected_bytes)
{
    std::vector<Job> jobs;
    for (const std::string& f : files) {
        Job j;
        j.path = f;
        j.original = mm::load_image(f);
        j.prepared = pre.process(j.original);
        if (j.prepared.input.size() != expected_bytes)
            throw std::runtime_error(net + ": preprocessed " + std::to_string(j.prepared.input.size()) +
                                     " bytes, network expects " + std::to_string(expected_bytes));
        jobs.push_back(std::move(j));
    }
    return jobs;
}

void* tag(size_t i) { return reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1)); }
size_t untag(void* p) { return static_cast<size_t>(reinterpret_cast<uintptr_t>(p)) - 1; }

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

}  // namespace

int main(int argc, char** argv)
{
    Args a;
    const int rc = parse(argc, argv, a);
    if (rc) return rc == 2 ? 0 : 1;
    for (const std::string& m : {a.det, a.cls}) {
        if (!file_exists(m)) {
            std::fprintf(stderr, "[error] model not found: %s\n        Download it from the DEEPX model zoo into models/ "
                                 "(see README.md), or pass --det / --cls.\n", m.c_str());
            return 1;
        }
    }

    // Callback state is declared before `inf` so it outlives the inference object.
    Lane det_lane, cls_lane;
    std::vector<Job> det_jobs, cls_jobs;

    try {
        // ---- 0. find the images first: a missing folder or too few images fails before the NPU is touched
        const std::vector<std::string> det_files = pick_images("det", a.det_dir, a.count);
        const std::vector<std::string> cls_files = pick_images("cls", a.cls_dir, a.count);

        // ---- 1. load both networks (all-or-nothing; OrtMode::Auto decides per model)
        NetworkOptions o;
        o.ort = a.ort;
        InferenceConfig cfg = config_for(ExecMode::Async);
        cfg.enable_stats = a.stats;
        CallbackInference inf({NetworkSpec("det", a.det, o), NetworkSpec("cls", a.cls, o)}, cfg);
        inf.print_info();

        // ---- 2. preprocess every image before feeding anything
        mm::Yolo26nFactory det_factory(a.score);
        mm::Resnet18Factory cls_factory(5);
        int dw = 0, dh = 0, cw = 0, ch = 0;
        input_hw(inf.info("det"), &dw, &dh);
        input_hw(inf.info("cls"), &cw, &ch);
        const auto t_pre = Clock::now();
        det_jobs = prepare("det", det_files, *det_factory.createPreprocessor(dw, dh), inf.input_size("det"));
        cls_jobs = prepare("cls", cls_files, *cls_factory.createPreprocessor(cw, ch), inf.input_size("cls"));
        std::printf("\npreprocessed %zu + %zu images in %.1f ms (%s: %s, %s: %s)\n", det_jobs.size(), cls_jobs.size(),
                    ms_since(t_pre), det_factory.getModelName().c_str(), det_factory.getTaskType().c_str(),
                    cls_factory.getModelName().c_str(), cls_factory.getTaskType().c_str());

        mm::make_dirs(a.out + "/det");
        mm::make_dirs(a.out + "/cls");
        det_lane.lines.resize(det_jobs.size());
        cls_lane.lines.resize(cls_jobs.size());

        // ---- 3. callbacks: run on each network's own delivery thread, one result at a time, in FIFO order
        auto det_post = det_factory.createPostprocessor(dw, dh);
        auto det_vis = det_factory.createVisualizer();
        inf.set_callback("det", [&](Result& r) {
            Lane& L = det_lane;
            const size_t i = untag(r.user_arg());
            if (r.seq() != L.received || i != L.received) ++L.order_errors;
            ++L.received;
            if (i >= det_jobs.size()) return;
            const Job& j = det_jobs[i];
            if (!r.ok()) { ++L.failed; L.lines[i] = mm::stem(j.path) + ": FAILED " + r.status().message(); return; }
            const mm::Detections boxes = det_post->process(r, j.prepared.lb);
            cv::Mat img = j.original.clone();
            det_vis->draw(img, boxes);
            mm::save_image(a.out + "/det/" + mm::stem(j.path) + "_overlay.jpg", img);
            ++L.saved;
            std::ostringstream s;
            s << mm::stem(j.path) << ": " << boxes.size() << " object(s)";
            for (size_t k = 0; k < boxes.size() && k < 6; ++k) {
                char b[64];
                std::snprintf(b, sizeof(b), "%s %.2f", mm::coco_name(boxes[k].class_id).c_str(), boxes[k].score);
                s << (k ? ", " : ": ") << b;
            }
            if (boxes.size() > 6) s << ", ...";
            L.lines[i] = s.str();
        });

        auto cls_post = cls_factory.createPostprocessor(cw, ch);
        auto cls_vis = cls_factory.createVisualizer();
        inf.set_callback("cls", [&](Result& r) {
            Lane& L = cls_lane;
            const size_t i = untag(r.user_arg());
            if (r.seq() != L.received || i != L.received) ++L.order_errors;
            ++L.received;
            if (i >= cls_jobs.size()) return;
            const Job& j = cls_jobs[i];
            if (!r.ok()) { ++L.failed; L.lines[i] = mm::stem(j.path) + ": FAILED " + r.status().message(); return; }
            const mm::TopKList top = cls_post->process(r, j.prepared.lb);
            cv::Mat img = j.original.clone();
            cls_vis->draw(img, top);
            mm::save_image(a.out + "/cls/" + mm::stem(j.path) + "_overlay.jpg", img);
            ++L.saved;
            std::ostringstream s;
            s << mm::stem(j.path) << ": ";
            for (size_t k = 0; k < top.size() && k < 3; ++k) {
                char b[96];
                std::snprintf(b, sizeof(b), "%s %.1f%%", mm::imagenet_name(top[k].class_id).c_str(), top[k].prob * 100.0);
                s << (k ? " | " : "") << b;
            }
            L.lines[i] = s.str();
        });

        // ---- 4. feed: one thread per network, every input back to back
        std::mutex err_m;
        std::string feed_error;
        double det_feed_ms = 0, cls_feed_ms = 0;
        auto feeder = [&](const std::string& net, const std::vector<Job>& jobs, double* feed_ms) {
            const auto t0 = Clock::now();
            try {
                for (size_t i = 0; i < jobs.size(); ++i)
                    inf.feed(net, jobs[i].prepared.input.data(), jobs[i].prepared.input.size(), tag(i));
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lk(err_m);
                feed_error += net + ": " + e.what() + "\n";
            }
            *feed_ms = ms_since(t0);
        };
        const auto t_run = Clock::now();
        std::thread td(feeder, "det", std::cref(det_jobs), &det_feed_ms);
        std::thread tc(feeder, "cls", std::cref(cls_jobs), &cls_feed_ms);
        td.join();
        tc.join();
        inf.wait_idle();
        const double run_ms = ms_since(t_run);
        if (!feed_error.empty()) std::fprintf(stderr, "[error] feeding failed:\n%s", feed_error.c_str());

        // ---- 5. summary (stdout + results.txt)
        std::ostringstream rep;
        rep << "\n=== results ===\n";
        auto section = [&](const char* net, const std::string& model, const Lane& L, size_t n, double feed_ms) {
            rep << "[" << net << "] " << model << "  (" << a.out << "/" << net << "/*_overlay.jpg)\n";
            for (const std::string& l : L.lines) rep << "  " << l << "\n";
            char b[256];
            std::snprintf(b, sizeof(b),
                          "  -> %zu images: received %llu, FIFO %s, failed %llu, saved %llu, all fed in %.1f ms, "
                          "callback errors %llu\n",
                          n, static_cast<unsigned long long>(L.received), L.order_errors ? "BROKEN" : "OK",
                          static_cast<unsigned long long>(L.failed), static_cast<unsigned long long>(L.saved), feed_ms,
                          static_cast<unsigned long long>(inf.callback_errors(net)));
            rep << b;
        };
        section("det", a.det, det_lane, det_jobs.size(), det_feed_ms);
        section("cls", a.cls, cls_lane, cls_jobs.size(), cls_feed_ms);
        char tot[160];
        const size_t total = det_jobs.size() + cls_jobs.size();
        std::snprintf(tot, sizeof(tot), "total: %zu results from 2 networks in %.1f ms (%.1f results/s)\n", total, run_ms,
                      run_ms > 0 ? total * 1000.0 / run_ms : 0.0);
        rep << tot;
        std::cout << rep.str();
        std::ofstream(a.out + "/results.txt") << rep.str();
        if (inf.stats_enabled()) inf.print_stats();

        const bool ok = feed_error.empty() &&
                        det_lane.received == det_jobs.size() && cls_lane.received == cls_jobs.size() &&
                        det_lane.order_errors + cls_lane.order_errors == 0 && det_lane.failed + cls_lane.failed == 0 &&
                        det_lane.saved == det_jobs.size() && cls_lane.saved == cls_jobs.size();
        std::cout << (ok ? "RESULT: PASS" : "RESULT: FAIL") << "  (check the overlay images by eye)\n";
        return ok ? 0 : 1;
    } catch (const InferenceError& e) {
        std::cerr << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
}