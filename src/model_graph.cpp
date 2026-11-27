// .dxnn task-graph reader + ORT decision. ALL .dxnn format knowledge is in this file.
//
// File layouts (mirrors dx_rt/lib/parsers/v7..v9_model_parser.cpp):
//   v7, v8 : "DXNN" | int32 LE version | JSON header, NUL-padded up to byte 8192 | data area ...
//   v9     : "DXNN" | u32 version | u64 header length | u32 crc | u32 flags | 8 reserved  (32-byte prelude)
//            | JSON header (header length bytes) | data area ...
//   JSON header: data.graph_info = {offset, size}, relative to the start of the data area.
//   Graph JSON : {"toposort_order": [...], "graphs": [{name, device: "NPU"|"CPU", head, tail,
//                 inputs: [{name, owner, users}], outputs: [{name, owner, users}]}]}
//   v6 uses an older graph schema (converted inside dx_rt) and is reported as Unknown.
// Anything unexpected -> GraphKind::Unknown + note; resolve_ort() then warns and enables ORT.
#include "dxinfer/model_graph.hpp"

#include "dxinfer/error.hpp"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace dxinfer {

namespace {

constexpr int64_t kV8HeaderEnd = 8192;
constexpr int64_t kV9PreludeSize = 32;
constexpr uint64_t kV9MaxHeader = 16ull * 1024 * 1024;
constexpr int64_t kMaxGraphJson = 64ll * 1024 * 1024;

ModelGraph unknown(ModelGraph g, const std::string& why)
{
    g.kind = GraphKind::Unknown;
    g.tasks.clear();
    g.npu_tasks = g.cpu_tasks = 0;
    g.cpu_head = g.cpu_tail = false;
    g.note = why;
    return g;
}

uint64_t le(const unsigned char* p, int n)
{
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | p[i];
    return v;
}

std::string json_error(const rapidjson::Document& d, const char* what)
{
    std::ostringstream os;
    os << what << " JSON parse error: " << rapidjson::GetParseError_En(d.GetParseError()) << " (at byte "
       << d.GetErrorOffset() << ")";
    return os.str();
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

void add_unique(std::vector<std::string>& v, const std::string& s)
{
    if (!s.empty() && std::find(v.begin(), v.end(), s) == v.end()) v.push_back(s);
}

bool read_at(std::ifstream& f, int64_t off, int64_t n, std::string& out)
{
    out.assign(static_cast<size_t>(n), '\0');
    f.clear();
    f.seekg(off);
    if (!f) return false;
    f.read(&out[0], n);
    return f.gcount() == n;
}

int64_t to_i64(const rapidjson::Value& v)
{
    if (v.IsInt64()) return v.GetInt64();
    if (v.IsUint64()) return static_cast<int64_t>(v.GetUint64());
    if (v.IsString()) return std::strtoll(v.GetString(), nullptr, 10);
    return -1;
}

std::string str_of(const rapidjson::Value& o, const char* key)
{
    return (o.HasMember(key) && o[key].IsString()) ? o[key].GetString() : std::string();
}

bool bool_of(const rapidjson::Value& o, const char* key)
{
    return o.HasMember(key) && o[key].IsBool() && o[key].GetBool();
}

// Parse the graph JSON into g (tasks in toposort order). Returns "" or an error.
std::string parse_graph(const std::string& json, ModelGraph& g)
{
    rapidjson::Document d;
    d.Parse(json.c_str(), json.size());
    if (d.HasParseError()) return json_error(d, "graph");
    if (!d.IsObject() || !d.HasMember("graphs") || !d["graphs"].IsArray())
        return "graph JSON has no 'graphs' array";

    std::vector<TaskNode> nodes;
    for (const auto& sg : d["graphs"].GetArray()) {
        if (!sg.IsObject()) return "graph JSON: 'graphs' entry is not an object";
        TaskNode t;
        t.name = str_of(sg, "name");
        if (t.name.empty()) return "graph JSON: subgraph without a name";
        const std::string dev = str_of(sg, "device");
        const std::string ldev = lower(dev);
        if (ldev == "npu") t.processor = Processor::Npu;
        else if (ldev == "cpu") t.processor = Processor::Cpu;
        else return "unknown device '" + dev + "' in subgraph '" + t.name + "'";
        t.head = bool_of(sg, "head");
        t.tail = bool_of(sg, "tail");
        if (sg.HasMember("inputs") && sg["inputs"].IsArray()) {
            for (const auto& in : sg["inputs"].GetArray()) {
                if (!in.IsObject()) continue;
                add_unique(t.inputs, str_of(in, "name"));
                const std::string owner = str_of(in, "owner");  // null owner = model input
                if (owner != t.name) add_unique(t.prevs, owner);
            }
        }
        if (sg.HasMember("outputs") && sg["outputs"].IsArray()) {
            for (const auto& out : sg["outputs"].GetArray()) {
                if (!out.IsObject()) continue;
                add_unique(t.outputs, str_of(out, "name"));
                if (out.HasMember("users") && out["users"].IsArray())
                    for (const auto& u : out["users"].GetArray())
                        if (u.IsString() && t.name != u.GetString()) add_unique(t.nexts, u.GetString());
            }
        }
        nodes.push_back(std::move(t));
    }

    std::vector<std::string> order;
    if (d.HasMember("toposort_order") && d["toposort_order"].IsArray()) {
        for (const auto& n : d["toposort_order"].GetArray())
            if (n.IsString()) order.emplace_back(n.GetString());
    } else {
        for (const auto& t : nodes) order.push_back(t.name);
    }
    for (const auto& name : order) {
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const TaskNode& t) { return t.name == name; });
        if (it == nodes.end()) return "toposort_order names '" + name + "', which is not in 'graphs'";
        g.tasks.push_back(*it);
    }
    if (g.tasks.size() != nodes.size()) return "toposort_order does not list every subgraph";
    return std::string();
}

void classify(ModelGraph& g)
{
    int first_npu = -1, last_npu = -1;
    for (size_t i = 0; i < g.tasks.size(); ++i) {
        if (g.tasks[i].processor == Processor::Npu) {
            ++g.npu_tasks;
            if (first_npu < 0) first_npu = static_cast<int>(i);
            last_npu = static_cast<int>(i);
        } else {
            ++g.cpu_tasks;
        }
    }
    for (size_t i = 0; i < g.tasks.size(); ++i) {
        if (g.tasks[i].processor != Processor::Cpu) continue;
        if (static_cast<int>(i) < first_npu) g.cpu_head = true;
        if (static_cast<int>(i) > last_npu) g.cpu_tail = true;
    }
    if (g.tasks.empty() || (g.npu_tasks == 1 && g.cpu_tasks == 0)) g.kind = GraphKind::NpuOnly;
    else if (g.npu_tasks == 1) g.kind = GraphKind::SingleNpuWithCpu;
    else if (g.npu_tasks >= 2) g.kind = GraphKind::MultiNpu;
}

std::string join(const std::vector<std::string>& v)
{
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
    return s;
}

const TaskNode* first_npu(const ModelGraph& g)
{
    for (const auto& t : g.tasks)
        if (t.processor == Processor::Npu) return &t;
    return nullptr;
}

/// "cpu_0 (tail), cpu_1 (head)"
std::string cpu_list(const ModelGraph& g)
{
    std::vector<std::string> v;
    bool seen_npu = false;
    for (const auto& t : g.tasks) {
        if (t.processor == Processor::Npu) { seen_npu = true; continue; }
        v.push_back(t.name + (seen_npu ? " (tail)" : " (head)"));
    }
    return join(v);
}

std::string capitalize(std::string s)
{
    if (!s.empty()) s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
    return s;
}

std::string prefix(const std::string& network) { return "[dxinfer] " + network + ": "; }

}  // namespace

// ------------------------------------------------------------------ reader

ModelGraph read_model_graph(const std::string& path) noexcept
{
    ModelGraph g;
    try {
        std::ifstream f(path, std::ios::binary);
        if (!f) return unknown(g, "cannot open '" + path + "'");
        f.seekg(0, std::ios::end);
        const int64_t file_size = static_cast<int64_t>(f.tellg());

        std::string pre;
        if (!read_at(f, 0, 8, pre)) return unknown(g, "truncated file (no 8-byte header)");
        if (pre.compare(0, 4, "DXNN") != 0) return unknown(g, "bad signature (not a .dxnn file)");
        const auto* p = reinterpret_cast<const unsigned char*>(pre.data());
        g.format_version = static_cast<int>(static_cast<int32_t>(le(p + 4, 4)));
        const int v = g.format_version;
        if (v == 6)
            return unknown(g, ".dxnn format v6 uses an older graph schema this reader does not parse (reader supports v" +
                                  std::to_string(kGraphReaderMinFormat) + "-v" + std::to_string(kGraphReaderMaxFormat) + ")");
        if (v < kGraphReaderMinFormat || v > kGraphReaderMaxFormat)
            return unknown(g, "unsupported .dxnn format v" + std::to_string(v) + " (reader supports v" +
                                  std::to_string(kGraphReaderMinFormat) + "-v" + std::to_string(kGraphReaderMaxFormat) + ")");

        std::string header;
        int64_t data_start = 0;
        if (v <= 8) {
            if (!read_at(f, 8, kV8HeaderEnd - 8, header)) return unknown(g, "truncated file (8 KB header incomplete)");
            header.resize(std::min(header.size(), header.find('\0')));  // JSON is NUL-padded
            data_start = kV8HeaderEnd;
        } else {
            std::string prelude;
            if (!read_at(f, 0, kV9PreludeSize, prelude)) return unknown(g, "truncated file (v9 prelude incomplete)");
            const uint64_t len = le(reinterpret_cast<const unsigned char*>(prelude.data()) + 8, 8);
            if (len == 0 || len > kV9MaxHeader)
                return unknown(g, "v9 header length " + std::to_string(len) + " is out of range");
            if (!read_at(f, kV9PreludeSize, static_cast<int64_t>(len), header))
                return unknown(g, "truncated file (v9 JSON header incomplete)");
            data_start = kV9PreludeSize + static_cast<int64_t>(len);
        }

        rapidjson::Document hd;
        hd.Parse(header.c_str(), header.size());
        if (hd.HasParseError()) return unknown(g, json_error(hd, "header"));
        if (!hd.IsObject() || !hd.HasMember("data") || !hd["data"].IsObject())
            return unknown(g, "header JSON has no 'data' object");
        const auto& data = hd["data"];
        if (!data.HasMember("graph_info") || !data["graph_info"].IsObject()) {
            classify(g);  // no graph info: dx-runtime runs the single NPU task
            return g;
        }
        const auto& gi = data["graph_info"];
        const int64_t off = gi.HasMember("offset") ? to_i64(gi["offset"]) : -1;
        const int64_t size = gi.HasMember("size") ? to_i64(gi["size"]) : -1;
        if (off < 0 || size < 0 || size > kMaxGraphJson)
            return unknown(g, "graph_info offset/size invalid (" + std::to_string(off) + ", " + std::to_string(size) + ")");
        if (size == 0) {
            classify(g);
            return g;
        }
        const int64_t at = data_start + off;
        std::string graph;
        if (at + size > file_size || !read_at(f, at, size, graph))
            return unknown(g, "truncated file: graph_info needs " + std::to_string(size) + " bytes at offset " +
                                  std::to_string(at) + ", file has " + std::to_string(file_size));
        const std::string err = parse_graph(graph, g);
        if (!err.empty()) return unknown(g, err);
        classify(g);
        if (g.kind == GraphKind::Unknown) return unknown(g, "graph has no NPU task");
        return g;
    } catch (const std::exception& e) {
        return unknown(g, std::string("reader error: ") + e.what());
    } catch (...) {
        return unknown(g, "reader error");
    }
}

std::string describe(const ModelGraph& g)
{
    if (g.kind == GraphKind::Unknown) return "unknown (" + g.note + ")";
    if (g.tasks.empty()) return "single NPU task";
    std::string s;
    for (const auto& t : g.tasks) s += (s.empty() ? "" : " -> ") + t.name + "[" + to_string(t.processor) + "]";
    return s;
}

// ------------------------------------------------------------------ decision

OrtDecision resolve_ort(const std::string& network, const std::string& path, OrtMode mode, const ModelGraph& g)
{
    OrtDecision d;
    switch (g.kind) {
    case GraphKind::NpuOnly:
        d.use_ort = (mode == OrtMode::On);
        break;

    case GraphKind::SingleNpuWithCpu: {
        // Short on purpose: counts only. print_info() lists the raw tensor names.
        const TaskNode* npu = first_npu(g);
        const std::string npu_name = npu ? npu->name : "the NPU task";
        std::vector<std::string> before, after, all;
        bool seen_npu = false;
        for (const auto& t : g.tasks) {
            if (t.processor == Processor::Npu) { seen_npu = true; continue; }
            (seen_npu ? after : before).push_back(t.name);
            all.push_back(t.name);
        }
        const std::string cpus = join(all);
        const bool several = all.size() > 1;
        std::string where;
        if (!before.empty()) where += join(before) + (before.size() > 1 ? " run" : " runs") + " on the CPU before " + npu_name;
        if (!after.empty())
            where += std::string(where.empty() ? "" : "; ") + join(after) + (after.size() > 1 ? " run" : " runs") +
                     " on the CPU after " + npu_name;
        std::string raw;
        if (npu && g.cpu_tail) raw += "results become " + npu_name + "'s " + std::to_string(npu->outputs.size()) + " raw tensors";
        if (npu && g.cpu_head)
            raw += std::string(raw.empty() ? "" : ", ") + "feed() takes " + npu_name + "'s " +
                   std::to_string(npu->inputs.size()) + " input tensors";
        const std::string ind = "\n          ";  // continuation lines align under "[dxinfer] "
        if (mode == OrtMode::Off) {
            d.use_ort = false;
            d.messages.push_back(prefix(network) + "ort=off: " + cpus + (several ? " are" : " is") +
                                 " not run, you implement " + (several ? "them" : "it") + " yourself." + ind +
                                 capitalize(raw) + "; listed by print_info().");
        } else {
            d.use_ort = true;
            if (mode == OrtMode::Auto)
                d.messages.push_back(prefix(network) + "ONNX Runtime on (ort=auto). " + where + "." + ind +
                                     "Faster option: ort=OrtMode::Off, then implement " + cpus + " yourself" + ind +
                                     "(" + raw + "; listed by print_info()).");
        }
        break;
    }

    case GraphKind::MultiNpu:
        d.required = true;
        if (mode == OrtMode::Off) {
            const TaskNode* npu = first_npu(g);
            throw InferenceError(
                ErrorCode::InvalidArgument, network,
                path + ": ort=off is not possible for this model: it has " + std::to_string(g.npu_tasks) +
                    " NPU tasks (" + describe(g) + "). dx-runtime without ONNX Runtime runs only the first NPU task (" +
                    (npu ? npu->name : std::string("?")) +
                    ") and never loads the others, so results would be wrong. Use OrtMode::Auto or OrtMode::On.");
        }
        d.use_ort = true;
        break;

    case GraphKind::Unknown:
        d.use_ort = (mode != OrtMode::Off);
        {
            std::string m = prefix(network) + "WARNING: could not read the task graph of " + path + " (" + g.note +
                            "). ORT: " + (d.use_ort ? "on" : "off") +
                            (mode == OrtMode::Auto ? " (safe default)" : " (as requested)") +
                            ". Graph-based checks are disabled for this network. This usually means the .dxnn "
                            "format changed: please update dxinfer's model_graph reader (src/model_graph.cpp).";
            if (mode == OrtMode::Off)
                m += " With ort=off, if this model has more than one NPU task its results are truncated to the "
                     "first NPU task.";
            d.messages.push_back(m);
        }
        break;
    }
    return d;
}

std::vector<std::string> check_loaded(const std::string& network, const OrtDecision& d, const ModelGraph& g,
                                      int loaded_tasks, bool ort_configured)
{
    std::vector<std::string> w;
    if (!d.use_ort || g.kind == GraphKind::Unknown || g.tasks.size() < 2) return w;
    const int expected = static_cast<int>(g.tasks.size());
    if (loaded_tasks <= 1) {
        const std::string why = std::string("dx-runtime loaded only ") + std::to_string(loaded_tasks) + " of " +
                                std::to_string(expected) + " tasks (ORT configured: " + (ort_configured ? "yes" : "no") +
                                "; dx_rt built without USE_ORT, or a 'debug' compile type)";
        if (d.required)
            throw InferenceError(ErrorCode::LoadFailed, network,
                                 "this model needs ONNX Runtime (" + describe(g) + ") but " + why +
                                     ". Results would be wrong.");
        const TaskNode* npu = first_npu(g);
        w.push_back(prefix(network) + "WARNING: " + why + ". CPU task(s) " + cpu_list(g) +
                    " are not run; results are " + (npu ? npu->name : std::string("the NPU task")) +
                    "'s raw tensors.");
        return w;
    }
    if (loaded_tasks != expected)
        w.push_back(prefix(network) + "WARNING: the .dxnn graph lists " + std::to_string(expected) +
                    " tasks but dx-runtime loaded " + std::to_string(loaded_tasks) +
                    ". dxinfer's model_graph reader may be out of date (src/model_graph.cpp).");
    return w;
}

}  // namespace dxinfer