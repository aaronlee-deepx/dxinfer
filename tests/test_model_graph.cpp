// Tests for the .dxnn task-graph reader and the ORT decision (no NPU needed).
#include "dxinfer/error.hpp"
#include "dxinfer/model_graph.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace dxinfer;

namespace {

std::string tmp_dir()
{
    static std::string dir;
    if (dir.empty()) {
        char tmpl[] = "/tmp/dxinfer_graph_XXXXXX";
        const char* d = mkdtemp(tmpl);
        dir = d ? d : "/tmp";
    }
    return dir;
}

void put_u32(std::string& s, uint32_t v) { for (int i = 0; i < 4; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xff)); }
void put_u64(std::string& s, uint64_t v) { for (int i = 0; i < 8; ++i) s.push_back(static_cast<char>((v >> (8 * i)) & 0xff)); }

std::string write_file(const std::string& name, const std::string& bytes)
{
    const std::string path = tmp_dir() + "/" + name;
    std::ofstream(path, std::ios::binary) << bytes;
    return path;
}

std::string header_json(size_t graph_offset, size_t graph_size)
{
    return "{\"data\":{\"graph_info\":{\"offset\":" + std::to_string(graph_offset) +
           ",\"size\":" + std::to_string(graph_size) + "}}}";
}

// v6..v8 layout: "DXNN" + int32 version + JSON header padded to 8192, then the data area.
std::string make_v8(const std::string& graph, int version = 8, size_t pad = 16)
{
    std::string f = "DXNN";
    put_u32(f, static_cast<uint32_t>(version));
    f += header_json(pad, graph.size());
    f.resize(8192, '\0');
    f += std::string(pad, 'x');  // something before graph_info inside the data area
    f += graph;
    return f;
}

// v9 layout: 32-byte prelude (sig, version, u64 header length, crc, flags, 8 reserved) + JSON header.
std::string make_v9(const std::string& graph, size_t pad = 8)
{
    const std::string hdr = header_json(pad, graph.size());
    std::string f = "DXNN";
    put_u32(f, 9);
    put_u64(f, hdr.size());
    put_u32(f, 0);  // crc (not checked by our reader)
    put_u32(f, 0);  // flags
    f += std::string(8, '\0');
    f += hdr;
    f += std::string(pad, 'x');
    f += graph;
    return f;
}

struct G {  // tiny graph-JSON builder: task(name, device, input-owner, head, tail)
    std::string order, graphs;
    G& task(const std::string& name, const std::string& dev, const std::string& in_owner, bool head, bool tail)
    {
        if (!order.empty()) { order += ","; graphs += ","; }
        order += "\"" + name + "\"";
        const std::string owner = in_owner.empty() ? "null" : "\"" + in_owner + "\"";
        const std::string in_name = in_owner.empty() ? "in_" + name : "out_" + in_owner;
        graphs += "{\"name\":\"" + name + "\",\"device\":\"" + dev + "\",\"inputs\":[{\"name\":\"" + in_name +
                  "\",\"owner\":" + owner + ",\"users\":[\"" + name + "\"]}],\"outputs\":[{\"name\":\"out_" + name +
                  "\",\"owner\":\"" + name + "\",\"users\":[]}],\"head\":" + (head ? "true" : "false") +
                  ",\"tail\":" + (tail ? "true" : "false") + "}";
        return *this;
    }
    std::string json() const { return "{\"toposort_order\":[" + order + "],\"graphs\":[" + graphs + "]}"; }
};

bool has_text(const std::vector<std::string>& msgs, const std::string& needle)
{
    for (const auto& m : msgs)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

size_t newlines(const std::string& s)
{
    size_t n = 0;
    for (char c : s) n += (c == '\n');
    return n;
}

// ------------------------------------------------------------------ reader

void v8_npu_then_cpu_tail()
{
    const auto g = read_model_graph(write_file("a.dxnn", make_v8(G().task("npu_0", "NPU", "", true, false)
                                                                     .task("cpu_0", "CPU", "npu_0", false, true)
                                                                     .json())));
    CHECK(g.kind == GraphKind::SingleNpuWithCpu);
    CHECK(g.format_version == 8);
    CHECK(g.tasks.size() == 2);
    CHECK(g.npu_tasks == 1 && g.cpu_tasks == 1);
    CHECK(g.cpu_tail && !g.cpu_head);
    CHECK(g.tasks.size() == 2 && g.tasks[0].name == "npu_0" && g.tasks[0].processor == Processor::Npu);
    CHECK(g.tasks.size() == 2 && g.tasks[1].processor == Processor::Cpu && g.tasks[1].prevs.size() == 1 &&
          g.tasks[1].prevs[0] == "npu_0");
    CHECK(g.tasks.size() == 2 && g.tasks[0].outputs.size() == 1 && g.tasks[0].outputs[0] == "out_npu_0");
    CHECK(describe(g) == "npu_0[NPU] -> cpu_0[CPU]");
}

void v9_interleaved_is_multi_npu()
{
    const auto g = read_model_graph(write_file("b.dxnn", make_v9(G().task("npu_0", "npu", "", true, false)
                                                                     .task("cpu_0", "cpu", "npu_0", false, false)
                                                                     .task("npu_1", "npu", "cpu_0", false, false)
                                                                     .task("cpu_1", "cpu", "npu_1", false, true)
                                                                     .json())));
    CHECK(g.kind == GraphKind::MultiNpu);
    CHECK(g.format_version == 9);
    CHECK(g.npu_tasks == 2 && g.cpu_tasks == 2);
}

void npu_npu_without_cpu_is_multi_npu()
{
    const auto g = read_model_graph(write_file("c.dxnn", make_v8(G().task("npu_0", "NPU", "", true, false)
                                                                     .task("npu_1", "NPU", "npu_0", false, true)
                                                                     .json(), 7)));
    CHECK(g.kind == GraphKind::MultiNpu);
    CHECK(g.cpu_tasks == 0);
}

void cpu_head_single_npu()
{
    const auto g = read_model_graph(write_file("d.dxnn", make_v9(G().task("cpu_0", "CPU", "", true, false)
                                                                     .task("npu_0", "NPU", "cpu_0", false, true)
                                                                     .json())));
    CHECK(g.kind == GraphKind::SingleNpuWithCpu);
    CHECK(g.cpu_head && !g.cpu_tail);
}

void single_npu_only()
{
    const auto g = read_model_graph(
        write_file("e.dxnn", make_v9(G().task("npu_0", "NPU", "", true, true).json())));
    CHECK(g.kind == GraphKind::NpuOnly);
}

void missing_graph_info_is_npu_only()
{
    std::string f = "DXNN";
    put_u32(f, 8);
    f += "{\"data\":{}}";
    f.resize(8192, '\0');
    const auto g = read_model_graph(write_file("f.dxnn", f));
    CHECK(g.kind == GraphKind::NpuOnly);
}

void failures_are_unknown_with_reason()
{
    const std::string good = G().task("npu_0", "NPU", "", true, false).task("cpu_0", "CPU", "npu_0", false, true).json();

    auto g = read_model_graph(tmp_dir() + "/does_not_exist.dxnn");
    CHECK(g.kind == GraphKind::Unknown && g.note.find("cannot open") != std::string::npos);

    std::string bad_sig = make_v8(good);
    bad_sig[0] = 'X';
    g = read_model_graph(write_file("g1.dxnn", bad_sig));
    CHECK(g.kind == GraphKind::Unknown && g.note.find("signature") != std::string::npos);

    std::string v10 = make_v9(good);
    v10[4] = 10;
    g = read_model_graph(write_file("g2.dxnn", v10));
    CHECK(g.kind == GraphKind::Unknown && g.note.find("v10") != std::string::npos);
    CHECK(g.format_version == 10);

    g = read_model_graph(write_file("g3.dxnn", make_v8(good).substr(0, 8192 + 20)));  // truncated graph
    CHECK(g.kind == GraphKind::Unknown && g.note.find("truncated") != std::string::npos);

    g = read_model_graph(write_file("g4.dxnn", make_v8("{\"graphs\": [ broken")));
    CHECK(g.kind == GraphKind::Unknown && g.note.find("JSON") != std::string::npos);

    g = read_model_graph(write_file("g5.dxnn", make_v8(G().task("npu_0", "XPU", "", true, true).json())));
    CHECK(g.kind == GraphKind::Unknown && g.note.find("XPU") != std::string::npos);

    g = read_model_graph(write_file("g6.dxnn", make_v8(good, 6)));  // v6 uses an older graph schema
    CHECK(g.kind == GraphKind::Unknown && g.note.find("v6") != std::string::npos);
}

void real_models_if_present()
{
    const char* paths[] = {
        "/home/user/deepx/dx-all-suite/dx-compiler/dx_com/output/YOLOV5S_Face-1.dxnn",
        "/home/user/deepx/dx-all-suite/dx-agent-dev-showcase/ultralytics-retrain-eval-deepx-export-ppe/"
        "yolo26n_deepx_model/yolo26n.dxnn",
    };
    for (const char* p : paths) {
        if (access(p, R_OK) != 0) {
            std::printf("  (skip: %s not found)\n", p);
            continue;
        }
        const auto g = read_model_graph(p);
        std::printf("  %s: v%d %s (%s)%s%s\n", p, g.format_version, describe(g).c_str(), to_string(g.kind),
                    g.note.empty() ? "" : " note: ", g.note.c_str());
        CHECK(g.kind == GraphKind::SingleNpuWithCpu);
        CHECK(g.tasks.size() == 2 && g.tasks[0].name == "npu_0" && g.tasks[1].name == "cpu_0");
        CHECK(g.cpu_tail);
    }
}

// ------------------------------------------------------------------ decision

ModelGraph kind(GraphKind k)
{
    ModelGraph g;
    g.kind = k;
    g.format_version = 9;
    TaskNode n{"npu_0", Processor::Npu, true, false, {}, {"cpu_0"}, {"in"}, {"raw0", "raw1"}};
    TaskNode c{"cpu_0", Processor::Cpu, false, true, {"npu_0"}, {}, {"raw0", "raw1"}, {"out"}};
    if (k == GraphKind::NpuOnly) { n.tail = true; n.nexts.clear(); g.tasks = {n}; g.npu_tasks = 1; }
    if (k == GraphKind::SingleNpuWithCpu) { g.tasks = {n, c}; g.npu_tasks = 1; g.cpu_tasks = 1; g.cpu_tail = true; }
    if (k == GraphKind::MultiNpu) {
        TaskNode n1{"npu_1", Processor::Npu, false, true, {"cpu_0"}, {}, {"out"}, {"final"}};
        c.tail = false;
        g.tasks = {n, c, n1};
        g.npu_tasks = 2;
        g.cpu_tasks = 1;
    }
    if (k == GraphKind::Unknown) g.note = "unsupported .dxnn format v10 (reader supports v7-v9)";
    return g;
}

void decide_npu_only()
{
    CHECK(!resolve_ort("n", "m.dxnn", OrtMode::Auto, kind(GraphKind::NpuOnly)).use_ort);
    CHECK(resolve_ort("n", "m.dxnn", OrtMode::Auto, kind(GraphKind::NpuOnly)).messages.empty());
    CHECK(resolve_ort("n", "m.dxnn", OrtMode::On, kind(GraphKind::NpuOnly)).use_ort);
    CHECK(!resolve_ort("n", "m.dxnn", OrtMode::Off, kind(GraphKind::NpuOnly)).use_ort);
}

void decide_single_npu_with_cpu()
{
    auto d = resolve_ort("n", "m.dxnn", OrtMode::Auto, kind(GraphKind::SingleNpuWithCpu));
    CHECK(d.use_ort && !d.required);
    CHECK(has_text(d.messages, "cpu_0"));
    CHECK(has_text(d.messages, "OrtMode::Off"));
    CHECK(has_text(d.messages, "2 raw tensors"));   // count only ...
    CHECK(has_text(d.messages, "print_info()"));    // ... the names are listed there
    CHECK(!has_text(d.messages, "raw0"));
    CHECK(d.messages.size() == 1 && newlines(d.messages[0]) <= 2);  // at most 3 lines

    d = resolve_ort("n", "m.dxnn", OrtMode::On, kind(GraphKind::SingleNpuWithCpu));
    CHECK(d.use_ort && d.messages.empty());

    d = resolve_ort("n", "m.dxnn", OrtMode::Off, kind(GraphKind::SingleNpuWithCpu));
    CHECK(!d.use_ort);
    CHECK(has_text(d.messages, "cpu_0"));  // tells the user what they now own
    CHECK(has_text(d.messages, "2 raw tensors"));
    CHECK(!has_text(d.messages, "raw0"));
    CHECK(d.messages.size() == 1 && newlines(d.messages[0]) <= 2);
}

void decide_multi_npu()
{
    auto d = resolve_ort("n", "m.dxnn", OrtMode::Auto, kind(GraphKind::MultiNpu));
    CHECK(d.use_ort && d.required);
    d = resolve_ort("n", "m.dxnn", OrtMode::On, kind(GraphKind::MultiNpu));
    CHECK(d.use_ort && d.required);
    CHECK_THROWS(resolve_ort("n", "m.dxnn", OrtMode::Off, kind(GraphKind::MultiNpu)), InferenceError);
    try {
        resolve_ort("n", "m.dxnn", OrtMode::Off, kind(GraphKind::MultiNpu));
    } catch (const InferenceError& e) {
        CHECK(e.code() == ErrorCode::InvalidArgument);
        CHECK(std::string(e.what()).find("npu_1") != std::string::npos);
    }
}

void decide_unknown_warns_in_every_mode()
{
    for (OrtMode m : {OrtMode::Auto, OrtMode::On, OrtMode::Off}) {
        const auto d = resolve_ort("n", "m.dxnn", m, kind(GraphKind::Unknown));
        CHECK(has_text(d.messages, "WARNING"));
        CHECK(has_text(d.messages, "v10"));          // the reason from the reader
        CHECK(has_text(d.messages, "model_graph"));  // tells us what to update
        CHECK(d.use_ort == (m != OrtMode::Off));
    }
    CHECK(has_text(resolve_ort("n", "m.dxnn", OrtMode::Off, kind(GraphKind::Unknown)).messages, "truncated"));
}

void after_load_checks()
{
    // ORT needed (2+ NPU tasks) but dx-runtime loaded a single task -> LoadFailed
    const auto multi = kind(GraphKind::MultiNpu);
    const auto dm = resolve_ort("n", "m.dxnn", OrtMode::Auto, multi);
    CHECK_THROWS(check_loaded("n", dm, multi, 1, false), InferenceError);
    try {
        check_loaded("n", dm, multi, 1, false);
    } catch (const InferenceError& e) {
        CHECK(e.code() == ErrorCode::LoadFailed);
        CHECK(std::string(e.what()).find("ONNX Runtime") != std::string::npos);
    }
    CHECK(check_loaded("n", dm, multi, 3, true).empty());  // all 3 tasks loaded: fine

    // single NPU + CPU tail, Auto, but dx-runtime dropped ORT -> warning only
    const auto single = kind(GraphKind::SingleNpuWithCpu);
    const auto ds = resolve_ort("n", "m.dxnn", OrtMode::Auto, single);
    CHECK(has_text(check_loaded("n", ds, single, 1, false), "raw"));
    CHECK(check_loaded("n", ds, single, 2, true).empty());

    // reader self-check: graph says 2 tasks, dx-runtime loaded 3 -> reader may be out of date
    CHECK(has_text(check_loaded("n", ds, single, 3, true), "model_graph"));

    // Off: dx-runtime loads exactly one task, nothing to report
    const auto doff = resolve_ort("n", "m.dxnn", OrtMode::Off, single);
    CHECK(check_loaded("n", doff, single, 1, false).empty());

    // Unknown graph: nothing to compare against
    const auto unk = kind(GraphKind::Unknown);
    CHECK(check_loaded("n", resolve_ort("n", "m.dxnn", OrtMode::Auto, unk), unk, 2, true).empty());
}

}  // namespace

int main()
{
    RUN(v8_npu_then_cpu_tail);
    RUN(v9_interleaved_is_multi_npu);
    RUN(npu_npu_without_cpu_is_multi_npu);
    RUN(cpu_head_single_npu);
    RUN(single_npu_only);
    RUN(missing_graph_info_is_npu_only);
    RUN(failures_are_unknown_with_reason);
    RUN(real_models_if_present);
    RUN(decide_npu_only);
    RUN(decide_single_npu_with_cpu);
    RUN(decide_multi_npu);
    RUN(decide_unknown_warns_in_every_mode);
    RUN(after_load_checks);
    return report("test_model_graph");
}