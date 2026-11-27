#include "dxinfer/inference_base.hpp"
#include "dxinfer/model_graph.hpp"

#include "network.hpp"

#include <iomanip>
#include <set>

namespace dxinfer {

namespace {

std::string shape_str(const std::vector<int64_t>& shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
    return s + "]";
}

void print_tensors(std::ostream& os, const char* title, const std::vector<TensorInfo>& ts, uint64_t total)
{
    os << "  " << std::left << std::setw(13) << title << ": " << ts.size() << " tensor(s), " << total
       << " bytes\n";
    for (size_t i = 0; i < ts.size(); ++i) {
        os << "     [" << i << "] " << std::setw(20) << ts[i].name << std::setw(18) << shape_str(ts[i].shape)
           << std::setw(7) << ts[i].dtype_name << std::right << std::setw(10) << ts[i].bytes << " B @"
           << ts[i].offset << std::left << "\n";
    }
    os << std::right;
}

const char* yes_no(bool b) { return b ? "yes" : "no"; }

// What a SingleNpuWithCpu model gives you with ort=off: the NPU task's raw tensors, one per line.
void print_ort_off(std::ostream& os, const ModelGraph& g)
{
    const TaskNode* npu = nullptr;
    std::string cpus;
    for (const TaskNode& t : g.tasks) {
        if (t.processor == Processor::Npu) { if (!npu) npu = &t; }
        else cpus += (cpus.empty() ? "" : ", ") + t.name;
    }
    if (!npu) return;
    if (g.cpu_tail) {
        os << "  with ort=off : results would be " << npu->name << "'s " << npu->outputs.size() << " raw tensors ("
           << cpus << " not run):\n";
        for (const std::string& n : npu->outputs) os << "                 " << n << "\n";
    }
    if (g.cpu_head) {
        os << "  with ort=off : feed() would take " << npu->name << "'s " << npu->inputs.size() << " input tensors:\n";
        for (const std::string& n : npu->inputs) os << "                 " << n << "\n";
    }
}

}  // namespace

InferenceBase::InferenceBase(const std::vector<NetworkSpec>& specs, const InferenceConfig& cfg) : cfg_(cfg)
{
    if (specs.empty()) throw InferenceError(ErrorCode::InvalidArgument, "", "no networks given");
    if (cfg.extra_result_buffers < -1)
        throw InferenceError(ErrorCode::InvalidArgument, "", "extra_result_buffers must be >= -1");

    // 1) validate every name before touching the device
    std::set<std::string> seen;
    for (const NetworkSpec& s : specs) {
        if (s.name.empty())
            throw InferenceError(ErrorCode::InvalidArgument, "", "network name must not be empty (" + s.path + ")");
        if (!seen.insert(s.name).second)
            throw InferenceError(ErrorCode::DuplicateName, s.name, "name is used for more than one network");
    }

    // 2) load all-or-nothing: on failure every network loaded so far is released
    nets_.reserve(specs.size());
    for (const NetworkSpec& s : specs) {
        try {
            std::unique_ptr<detail::Network> n(new detail::Network(s, cfg_));
            by_name_[s.name] = n.get();
            nets_.push_back(std::move(n));
        } catch (const InferenceError& e) {
            std::string loaded;
            for (const auto& n : nets_) loaded += (loaded.empty() ? "" : ", ") + n->name();
            nets_.clear();  // releases the engines -> NPU memory freed
            by_name_.clear();
            throw InferenceError(e.code(), e.network(),
                                 e.message() + (loaded.empty() ? std::string()
                                                               : " [already loaded and now released: " + loaded + "]"));
        }
    }
}

InferenceBase::~InferenceBase() { close_all(); }

void InferenceBase::close_all() noexcept
{
    for (auto& n : nets_) {
        try { n->close(); } catch (...) {}
    }
}

detail::Network& InferenceBase::network(const std::string& name) const
{
    auto it = by_name_.find(name);
    if (it == by_name_.end()) {
        std::string known;
        for (const auto& n : nets_) known += (known.empty() ? "" : ", ") + n->name();
        throw InferenceError(ErrorCode::UnknownNetwork, name, "no such network (known: " + known + ")");
    }
    return *it->second;
}

// ---------------------------------------------------------------- queries

std::vector<std::string> InferenceBase::network_names() const
{
    std::vector<std::string> names;
    for (const auto& n : nets_) names.push_back(n->name());
    return names;
}

bool InferenceBase::has_network(const std::string& name) const { return by_name_.count(name) != 0; }
const NetworkInfo& InferenceBase::info(const std::string& name) const { return network(name).info(); }
uint64_t InferenceBase::input_size(const std::string& name) const { return info(name).input_bytes; }
uint64_t InferenceBase::output_size(const std::string& name) const { return info(name).output_bytes; }

const std::vector<TensorInfo>& InferenceBase::input_tensors(const std::string& name) const
{
    return info(name).inputs;
}

const std::vector<TensorInfo>& InferenceBase::output_tensors(const std::string& name) const
{
    return info(name).outputs;
}

void InferenceBase::print_info(const std::string& name, std::ostream& os) const
{
    const NetworkInfo& i = info(name);
    os << "network '" << i.name << "'  (" << i.path << ")\n";
    os << "  model        : " << i.model_name << "  version " << i.model_version << "  compile " << i.compile_type
       << "\n";
    os << "  flags        : PPU " << yes_no(i.ppu) << " | multi-input "
       << yes_no(i.multi_input) << " | dynamic output " << yes_no(i.dynamic_output) << "\n";
    os << "  task graph   : " << describe(i.graph) << "  (" << to_string(i.graph.kind);
    if (i.graph.format_version) os << ", .dxnn v" << i.graph.format_version;
    os << ")\n";
    os << "  ONNX Runtime : requested " << to_string(i.ort_requested) << " -> " << (i.ort ? "on" : "off")
       << (i.ort_required ? " (required)" : "") << ", dx-runtime loaded " << i.loaded_tasks << " task(s)\n";
    if (i.graph.kind == GraphKind::SingleNpuWithCpu) print_ort_off(os, i.graph);
    os << "  exec mode    : " << to_string(i.exec) << "\n";
    print_tensors(os, "inputs", i.inputs, i.input_bytes);
    print_tensors(os, "outputs", i.outputs, i.output_bytes);
    os << "  device slots : " << i.buffer_count << " buffers x " << i.device_count << " device(s) = "
       << i.max_in_flight << " in flight, " << i.result_buffers << " result blocks\n";
}

void InferenceBase::print_info(std::ostream& os) const
{
    for (const auto& n : nets_) print_info(n->name(), os);
}

// ---------------------------------------------------------------- feeding

uint64_t InferenceBase::feed(const std::string& name, const void* input, size_t bytes, void* user_arg)
{
    detail::Network& n = network(name);
    check_can_feed(n);
    uint64_t seq = 0;
    n.submit(input, bytes, user_arg, true, &seq);
    return seq;
}

bool InferenceBase::try_feed(const std::string& name, const void* input, size_t bytes, void* user_arg,
                             uint64_t* seq)
{
    detail::Network& n = network(name);
    check_can_feed(n);
    return n.submit(input, bytes, user_arg, false, seq);
}

void InferenceBase::wait_completed()
{
    for (auto& n : nets_) n->wait_completed();
}

uint64_t InferenceBase::pending(const std::string& name) const { return network(name).pending(); }

// ---------------------------------------------------------------- statistics

NetworkStats InferenceBase::stats(const std::string& name) const { return network(name).stats(); }
void InferenceBase::reset_stats(const std::string& name) { network(name).reset_stats(); }

void InferenceBase::reset_stats()
{
    for (auto& n : nets_) n->reset_stats();
}

void InferenceBase::print_stats(const std::string& name, std::ostream& os) const
{
    dxinfer::print_stats(os, stats(name));
}

void InferenceBase::print_stats(std::ostream& os) const
{
    for (const auto& n : nets_) dxinfer::print_stats(os, n->stats());
}

}  // namespace dxinfer