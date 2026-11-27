#include "network.hpp"

#include "dxinfer/model_graph.hpp"
#include "dxinfer/pool_warning.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace dxinfer {
namespace detail {

namespace {

constexpr int kDefaultBufferCount = 6;  // dx-runtime default (DXRT_TASK_MAX_LOAD)
constexpr int kMaxBufferCount = 100;
constexpr size_t kPoolLowMark = 1;      // warn when free result blocks drop to this
constexpr size_t kPoolRecoverMark = 3;  // "recovered" when back to this

int effective_buffer_count(int requested)
{
    if (requested > 0) return requested;
    if (const char* env = std::getenv("DXRT_TASK_MAX_LOAD")) {
        const int v = std::atoi(env);
        if (v > 0) return v;
    }
    return kDefaultBufferCount;
}

int effective_device_count(const std::vector<int>& devices)
{
    if (!devices.empty()) return static_cast<int>(devices.size());
    return std::max(1, dxrt::DeviceStatus::GetDeviceCount());
}

bool mentions_memory(const char* msg)
{
    std::string s(msg ? msg : "");
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s.find("memory") != std::string::npos;
}

std::string free_memory_report(const std::vector<int>& devices)
{
    std::vector<int> ids = devices;
    if (ids.empty()) {
        int n = 0;
        try { n = dxrt::DeviceStatus::GetDeviceCount(); } catch (const std::exception&) {}
        for (int i = 0; i < n; ++i) ids.push_back(i);
    }
    std::string out;
    for (int id : ids) {
        try {
            const uint64_t free_b = dxrt::DeviceStatus::GetCurrentStatus(id).GetMemoryFree();
            out += "; device " + std::to_string(id) + " free " + std::to_string(free_b / (1024 * 1024)) + " MiB";
        } catch (const std::exception&) {
        }
    }
    return out;
}

// dx-runtime load failure -> InferenceError (OutOfDeviceMemory when the NPU is full)
InferenceError load_error(const NetworkSpec& spec, const dxrt::Exception& e)
{
    if (e.code() == dxrt::ERROR_CODE::INSUFFICIENT_MEMORY || mentions_memory(e.what())) {
        return InferenceError(ErrorCode::OutOfDeviceMemory, spec.name,
                              spec.path + ": not enough NPU memory to load this network together with the "
                                          "networks loaded before it (" + e.what() + ")" +
                                  free_memory_report(spec.options.devices));
    }
    return InferenceError(ErrorCode::LoadFailed, spec.name,
                          spec.path + ": " + dxrt::to_string(e.code()) + ": " + e.what());
}

TensorInfo to_info(const dxrt::Tensor& t, uint64_t offset)
{
    TensorInfo ti;
    ti.name = t.name();
    ti.dtype = static_cast<int>(t.type());
    ti.dtype_name = dxrt::DataTypeToString(t.type());
    ti.shape = t.shape();
    ti.bytes = t.size_in_bytes();
    ti.offset = offset;
    return ti;
}

}  // namespace

// ------------------------------------------------------------------ setup

Network::Network(const NetworkSpec& spec, const InferenceConfig& cfg)
    : name_(std::make_shared<const std::string>(spec.name)), exec_(cfg.exec), stats_(cfg.enable_stats)
{
    const NetworkOptions& o = spec.options;
    if (o.bound < 0 || o.bound > 6)
        throw InferenceError(ErrorCode::InvalidArgument, spec.name, "bound must be 0..6");
    if (o.buffer_count < 0 || o.buffer_count > kMaxBufferCount)
        throw InferenceError(ErrorCode::InvalidArgument, spec.name, "buffer_count must be 0..100");

    // Read the task graph from the .dxnn first: dx-runtime without ORT runs only the
    // first NPU task, so decide here (may throw InvalidArgument for ort=off on 2+ NPU tasks).
    const ModelGraph graph = read_model_graph(spec.path);
    const OrtDecision ort = resolve_ort(spec.name, spec.path, o.ort, graph);
    for (const std::string& m : ort.messages) std::fprintf(stderr, "%s\n", m.c_str());

    dxrt::InferenceOption opt;
    opt.bufferCount = o.buffer_count;
    opt.boundOption = static_cast<uint32_t>(o.bound);
    opt.devices = o.devices;
    opt.useORT = ort.use_ort;
    opt.showModelInfo = false;
    try {
        engine_.reset(new dxrt::InferenceEngine(spec.path, opt));
    } catch (const dxrt::Exception& e) {
        throw load_error(spec, e);
    } catch (const std::exception& e) {
        throw InferenceError(ErrorCode::LoadFailed, spec.name, spec.path + ": " + e.what());
    }

    build_info(spec, cfg);
    info_.graph = graph;
    info_.ort_requested = o.ort;
    info_.ort_required = ort.required;
    try {
        info_.loaded_tasks = static_cast<int>(engine_->GetTaskOrder().size());
    } catch (const std::exception&) {
        info_.loaded_tasks = 0;
    }
    // What dx-runtime really loaded vs the graph (throws LoadFailed if ORT was needed but dropped).
    for (const std::string& m : check_loaded(spec.name, ort, graph, info_.loaded_tasks, info_.ort))
        std::fprintf(stderr, "%s\n", m.c_str());

    pool_ = BufferPool::create(static_cast<size_t>(info_.output_bytes), static_cast<size_t>(info_.result_buffers));
    // Low result-buffer warning (stderr). Only meaningful with spare blocks: with
    // extra_result_buffers < 2 the pool runs near empty by design. The observer
    // captures only the warner, so it is safe if Results outlive this network.
    if (info_.result_buffers >= info_.max_in_flight + 2) {
        auto warner = std::make_shared<PoolWarner>(spec.name, std::cerr, std::chrono::milliseconds(1000),
                                                   info_.result_buffers - info_.max_in_flight);
        auto occ = occ_;
        pool_->set_level_watch(kPoolLowMark, kPoolRecoverMark,
                               std::make_shared<const BufferPool::LevelFn>(
                                   [warner, occ](bool low, size_t free_blocks, size_t total) {
                                       PoolOccupancy now;
                                       now.in_flight = occ->in_flight.load(std::memory_order_relaxed);
                                       now.waiting = occ->waiting.load(std::memory_order_relaxed);
                                       const size_t used = total > free_blocks ? total - free_blocks : 0;
                                       const size_t known = now.in_flight + now.waiting;
                                       now.held_by_app = used > known ? used - known : 0;
                                       warner->on_level(low, free_blocks, total, now);
                                   }));
    }
    jobs_.resize(static_cast<size_t>(info_.max_in_flight));
    for (Job& j : jobs_) free_jobs_.push_back(&j);

    if (exec_ == ExecMode::Async) {
        // Runs on a dx-runtime thread: keep it short (complete() only stores the result).
        engine_->RegisterCallback([this](dxrt::TensorPtrs& outputs, void* arg) -> int {
            try {
                complete(static_cast<Job*>(arg), &outputs, Status());
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[dxinfer] %s: completion handling failed: %s\n", name().c_str(), e.what());
            }
            return 0;
        });
    }
}

void Network::build_info(const NetworkSpec& spec, const InferenceConfig& cfg)
{
    try {
        info_.name = spec.name;
        info_.path = spec.path;
        info_.model_name = engine_->GetModelName();
        info_.model_version = engine_->GetModelVersion();
        info_.compile_type = engine_->GetCompileType();
        info_.ppu = engine_->IsPPU();
        info_.ort = engine_->IsOrtConfigured();  // effective: dx-runtime forces it off without USE_ORT / debug
        info_.multi_input = engine_->IsMultiInputModel();
        info_.dynamic_output = engine_->HasDynamicOutput();

        uint64_t off = 0;
        for (const dxrt::Tensor& t : engine_->GetInputs()) {
            info_.inputs.push_back(to_info(t, off));
            off += t.size_in_bytes();
        }
        info_.input_bytes = engine_->GetInputSize();
        if (info_.multi_input) {
            input_parts_ = engine_->GetInputTensorSizes();
            uint64_t sum = 0;
            for (uint64_t s : input_parts_) sum += s;
            info_.input_bytes = sum;  // feed() takes all input tensors back to back
        }

        off = 0;
        for (const dxrt::Tensor& t : engine_->GetOutputs()) {
            uint64_t at = off;
            try { at = engine_->GetOutputTensorOffset(t.name()); } catch (const std::exception&) {}
            info_.outputs.push_back(to_info(t, at));
            off += t.size_in_bytes();
        }
        info_.output_bytes = engine_->GetOutputSize();

        info_.buffer_count = effective_buffer_count(spec.options.buffer_count);
        info_.device_count = effective_device_count(spec.options.devices);
        info_.max_in_flight = info_.buffer_count * info_.device_count;
        const int extra = cfg.extra_result_buffers < 0 ? info_.max_in_flight : cfg.extra_result_buffers;
        info_.result_buffers = info_.max_in_flight + extra;
        info_.exec = cfg.exec;
    } catch (const dxrt::Exception& e) {
        throw InferenceError(ErrorCode::LoadFailed, spec.name, std::string("reading model info: ") + e.what());
    }
}

Network::~Network()
{
    close();
    if (exec_ == ExecMode::Async) {
        try { engine_->RegisterCallback(nullptr); } catch (const std::exception&) {}
    }
}

void Network::close()
{
    pool_->set_level_watch(0, 0, nullptr);  // no warnings during shutdown or for Results kept afterwards
    {
        std::lock_guard<std::mutex> lk(slot_m_);
        closing_ = true;
    }
    slot_cv_.notify_all();  // feeders waiting for a slot -> ShuttingDown
    pool_->shutdown();      // feeders waiting for a block  -> ShuttingDown
    {
        std::unique_lock<std::mutex> lk(slot_m_);
        slot_cv_.wait(lk, [this] { return free_jobs_.size() == jobs_.size(); });  // in-flight jobs done
    }
    results_.close();
}

// ------------------------------------------------------------------ feeding

Network::Job* Network::acquire_job(bool wait, int* in_flight)
{
    std::unique_lock<std::mutex> lk(slot_m_);
    if (wait) slot_cv_.wait(lk, [this] { return !free_jobs_.empty() || closing_; });
    if (closing_) throw InferenceError(ErrorCode::ShuttingDown, name(), "network is closing");
    if (free_jobs_.empty()) return nullptr;
    Job* job = free_jobs_.back();
    free_jobs_.pop_back();
    occ_->in_flight.fetch_add(1, std::memory_order_relaxed);
    *in_flight = static_cast<int>(jobs_.size() - free_jobs_.size());
    return job;
}

void Network::release_job(Job* job)
{
    {
        std::lock_guard<std::mutex> lk(slot_m_);
        free_jobs_.push_back(job);
        occ_->in_flight.fetch_sub(1, std::memory_order_relaxed);
    }
    slot_cv_.notify_all();  // a feeder (and close()) may be waiting
}

std::vector<void*> Network::split_input(void* input) const
{
    std::vector<void*> parts;
    uint8_t* p = static_cast<uint8_t*>(input);
    for (uint64_t sz : input_parts_) {
        parts.push_back(p);
        p += sz;
    }
    return parts;
}

bool Network::submit(const void* input, size_t bytes, void* user_arg, bool wait, uint64_t* seq_out)
{
    if (!input) throw InferenceError(ErrorCode::InvalidArgument, name(), "input is null");
    if (bytes != info_.input_bytes)
        throw InferenceError(ErrorCode::BadInputSize, name(),
                             "got " + std::to_string(bytes) + " bytes, network expects " +
                                 std::to_string(info_.input_bytes));

    int in_flight = 0;
    Job* job = acquire_job(wait, &in_flight);
    if (!job) return false;
    job->buffer = wait ? pool_->acquire() : pool_->try_acquire();
    if (!job->buffer) {  // all result blocks are held by results not yet consumed
        release_job(job);
        if (wait) throw InferenceError(ErrorCode::ShuttingDown, name(), "network is closing");
        return false;
    }

    job->seq = next_seq_.fetch_add(1);
    job->user_arg = user_arg;
    job->times = JobTimes();
    if (seq_out) *seq_out = job->seq;
    {
        std::lock_guard<std::mutex> lk(count_m_);
        ++fed_;
    }
    stats_.on_feed(job->times, in_flight);

    // dx-runtime copies the input during submission -> caller may reuse it after we return.
    void* in = const_cast<void*>(input);
    const bool multi = input_parts_.size() > 1;
    dxrt::TensorPtrs outputs;
    Status status;
    bool done_here = false;  // async success: completion arrives on the dxrt callback thread
    try {
        if (exec_ == ExecMode::Async) {
            // `job` must not be touched after a successful RunAsync (it may already be completed).
            if (multi) engine_->RunAsync(split_input(in), job, job->buffer.data());
            else       engine_->RunAsync(in, job, job->buffer.data());
        } else {
            outputs = multi ? engine_->RunMultiInput(split_input(in), nullptr, job->buffer.data())
                            : engine_->Run(in, nullptr, job->buffer.data());
            done_here = true;
        }
    } catch (const std::exception& e) {
        // Still deliver a Result for this seq (with the error), so FIFO never stalls.
        status = Status(ErrorCode::InferenceFailed, e.what());
        done_here = true;
    }
    if (done_here) complete(job, status.ok() ? &outputs : nullptr, std::move(status));
    return true;
}

// ------------------------------------------------------------------ completion

void Network::complete(Job* job, const dxrt::TensorPtrs* outputs, Status status)
{
    std::vector<TensorView> views;
    if (status.ok()) views = make_views(job->buffer.data(), job->buffer.size(), outputs, status);
    stats_.on_complete(job->times, status.ok());

    const uint64_t seq = job->seq;
    Result r = ResultAccess::make(name_, seq, job->user_arg, std::move(status), std::move(job->buffer),
                                  std::move(views), info_.output_bytes, job->times);
    occ_->waiting.fetch_add(1, std::memory_order_relaxed);  // before release: the block is never "held by app"
    release_job(job);                  // free the slot first: the device gets refilled right away
    results_.push(seq, std::move(r));  // FIFO release happens inside the reorder buffer
    {
        std::lock_guard<std::mutex> lk(count_m_);
        ++completed_;
    }
    count_cv_.notify_all();
}

// Build tensor views over the result block. Normally dx-runtime has written
// every output into our block (zero-copy). Anything it left in its own memory
// (only valid during the callback) is copied into the block behind the rest.
std::vector<TensorView> Network::make_views(uint8_t* base, size_t capacity, const dxrt::TensorPtrs* outputs,
                                            Status& status) const
{
    std::vector<TensorView> views;
    if (!outputs || outputs->empty()) {  // no runtime metadata: use the static layout
        for (const TensorInfo& t : info_.outputs) {
            TensorView v;
            v.name = t.name;
            v.dtype = t.dtype;
            v.shape = t.shape;
            v.data = base + t.offset;
            v.bytes = t.bytes;
            views.push_back(std::move(v));
        }
        return views;
    }

    const uintptr_t lo = reinterpret_cast<uintptr_t>(base);
    const uintptr_t hi = lo + capacity;
    std::vector<const uint8_t*> outside(outputs->size(), nullptr);
    size_t used_end = 0;
    views.resize(outputs->size());
    for (size_t i = 0; i < outputs->size(); ++i) {
        const dxrt::Tensor& t = *(*outputs)[i];
        const uint8_t* p = static_cast<const uint8_t*>(t.data());
        const size_t n = t.size_in_bytes();
        views[i].name = t.name();
        views[i].dtype = static_cast<int>(t.type());
        views[i].shape = t.shape();
        views[i].bytes = n;
        const uintptr_t a = reinterpret_cast<uintptr_t>(p);
        if (p && a >= lo && a + n <= hi) {
            views[i].data = p;
            used_end = std::max(used_end, static_cast<size_t>(a - lo) + n);
        } else {
            outside[i] = p;
        }
    }
    for (size_t i = 0; i < outputs->size(); ++i) {
        if (!outside[i]) continue;
        const size_t n = static_cast<size_t>(views[i].bytes);
        if (used_end + n > capacity) {
            status = Status(ErrorCode::InferenceFailed,
                            "output tensor '" + views[i].name + "' does not fit into the result block");
            return {};
        }
        std::memcpy(base + used_end, outside[i], n);
        views[i].data = base + used_end;
        used_end += n;
    }
    return views;
}

// ------------------------------------------------------------------ bookkeeping

void Network::on_consumed()
{
    {
        std::lock_guard<std::mutex> lk(count_m_);
        ++consumed_;
    }
    count_cv_.notify_all();
}

void Network::wait_completed()
{
    std::unique_lock<std::mutex> lk(count_m_);
    count_cv_.wait(lk, [this] { return completed_ == fed_; });
}

void Network::wait_consumed()
{
    std::unique_lock<std::mutex> lk(count_m_);
    count_cv_.wait(lk, [this] { return consumed_ == fed_; });
}

uint64_t Network::pending() const
{
    std::lock_guard<std::mutex> lk(count_m_);
    return fed_ - consumed_;
}

NetworkStats Network::stats() const
{
    NetworkStats s = stats_.snapshot();
    s.network = name();
    s.max_in_flight = info_.max_in_flight;
    if (s.enabled) {
        try {
            s.npu_us = engine_->GetNpuInferenceTimeMean();
            s.queue_wait_us = engine_->GetQueueWaitTimeMean();
            s.dxrt_e2e_us = engine_->GetE2EResponseTimeMean();
            s.core_latency_us = engine_->GetCoreLatencyMean();
            s.core_samples = engine_->GetCoreLatencyCnt();
            s.has_engine_counters = true;
        } catch (const std::exception&) {
            s.has_engine_counters = false;
        }
    }
    return s;
}

}  // namespace detail
}  // namespace dxinfer