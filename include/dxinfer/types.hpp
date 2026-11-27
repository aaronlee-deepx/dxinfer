// Plain configuration / description types shared by all dxinfer classes.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dxinfer {

/// How dx-runtime is driven underneath. Independent of how results are
/// delivered to you (BlockingInference vs CallbackInference).
enum class ExecMode {
    Sync,   ///< InferenceEngine::Run() on the feeding thread (one job per feeding thread)
    Async,  ///< InferenceEngine::RunAsync() + dx-runtime completion callback (keeps the device full)
};

inline const char* to_string(ExecMode m) noexcept
{
    return m == ExecMode::Sync ? "sync (Run)" : "async (RunAsync + completion callback)";
}

/// Whether the CPU (ONNX Runtime) tasks of a model run.
/// dx-runtime with ORT off loads ONLY the first NPU task: CPU tasks are skipped and later
/// NPU tasks are never loaded. Auto reads the model's task graph and decides (see README).
enum class OrtMode {
    Auto,  ///< decide from the model's task graph (default)
    On,    ///< always run every task (NPU + CPU)
    Off,   ///< first NPU task only; you implement the CPU part yourself. Rejected for models with 2+ NPU tasks.
};

inline const char* to_string(OrtMode m) noexcept
{
    switch (m) {
    case OrtMode::Auto: return "auto";
    case OrtMode::On:   return "on";
    case OrtMode::Off:  return "off";
    }
    return "?";
}

/// Per-network dx-runtime options (mirror dxrt::InferenceOption and dxrun flags).
struct NetworkOptions {
    int buffer_count = 0;          ///< dxrt I/O buffers per device; 0 = $DXRT_TASK_MAX_LOAD or 6
    std::vector<int> devices;      ///< device ids to use; empty = all devices
    int bound = 0;                 ///< NPU core binding 0..6 (NPU_ALL, NPU_0, NPU_1, NPU_2, NPU_01, NPU_12, NPU_02)
    OrtMode ort = OrtMode::Auto;   ///< CPU (ONNX Runtime) tasks; see OrtMode
};

/// One network to load: a unique name you choose + the .dxnn path.
struct NetworkSpec {
    std::string name;
    std::string path;
    NetworkOptions options;

    NetworkSpec(std::string n, std::string p, NetworkOptions o = NetworkOptions())
        : name(std::move(n)), path(std::move(p)), options(std::move(o)) {}
};

/// Options that apply to the whole inference object.
struct InferenceConfig {
    ExecMode exec = ExecMode::Async;
    bool enable_stats = false;      ///< collect per-network FPS / latency statistics
    int extra_result_buffers = -1;  ///< output blocks beyond max-in-flight, for results you still hold;
                                    ///< -1 = same as max-in-flight
};

inline InferenceConfig config_for(ExecMode m)
{
    InferenceConfig c;
    c.exec = m;
    return c;
}

/// Layout of one input or output tensor.
struct TensorInfo {
    std::string name;
    int dtype = 0;               ///< dxrt::DataType value
    std::string dtype_name;      ///< "UINT8", "FLOAT", ...
    std::vector<int64_t> shape;
    uint64_t bytes = 0;
    uint64_t offset = 0;         ///< byte offset inside the network's input / output block
};

// ------------------------------------------------------------ model task graph

enum class Processor { Npu, Cpu };

inline const char* to_string(Processor p) noexcept { return p == Processor::Npu ? "NPU" : "CPU"; }

/// One task (subgraph) of a compiled model, as stored in the .dxnn graph info.
struct TaskNode {
    std::string name;                  ///< e.g. "npu_0", "cpu_0"
    Processor processor = Processor::Npu;
    bool head = false;                 ///< receives model inputs
    bool tail = false;                 ///< produces model outputs
    std::vector<std::string> prevs;    ///< tasks whose outputs this task consumes
    std::vector<std::string> nexts;    ///< tasks that consume this task's outputs
    std::vector<std::string> inputs;   ///< input tensor names
    std::vector<std::string> outputs;  ///< output tensor names
};

enum class GraphKind {
    Unknown,           ///< the graph could not be read (see ModelGraph::note)
    NpuOnly,           ///< a single NPU task: ORT makes no difference
    SingleNpuWithCpu,  ///< one NPU task plus CPU task(s) before and/or after it
    MultiNpu,          ///< two or more NPU tasks: ORT is required to run the whole model
};

inline const char* to_string(GraphKind k) noexcept
{
    switch (k) {
    case GraphKind::Unknown:          return "Unknown";
    case GraphKind::NpuOnly:          return "NpuOnly";
    case GraphKind::SingleNpuWithCpu: return "SingleNpuWithCpu";
    case GraphKind::MultiNpu:         return "MultiNpu";
    }
    return "?";
}

struct ModelGraph {
    GraphKind kind = GraphKind::Unknown;
    int format_version = 0;        ///< .dxnn file format version (0 if not read)
    std::vector<TaskNode> tasks;   ///< in execution (toposort) order
    int npu_tasks = 0;
    int cpu_tasks = 0;
    bool cpu_head = false;         ///< a CPU task runs before the first NPU task
    bool cpu_tail = false;         ///< a CPU task runs after the last NPU task
    std::string note;              ///< why kind == Unknown (empty otherwise)
};

/// Everything known about a loaded network.
struct NetworkInfo {
    std::string name;
    std::string path;
    std::string model_name;
    std::string model_version;
    std::string compile_type;
    bool ppu = false;
    bool ort = false;              ///< effective: dx-runtime really runs the CPU / later NPU tasks
    OrtMode ort_requested = OrtMode::Auto;
    bool ort_required = false;     ///< the model cannot run correctly without ORT (2+ NPU tasks)
    ModelGraph graph;              ///< task graph read from the .dxnn before loading
    int loaded_tasks = 0;          ///< tasks dx-runtime actually loaded (GetTaskOrder().size())
    bool multi_input = false;
    bool dynamic_output = false;
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
    uint64_t input_bytes = 0;    ///< bytes feed() expects (all inputs back to back)
    uint64_t output_bytes = 0;   ///< bytes of one result block
    int buffer_count = 0;        ///< dxrt buffers per device
    int device_count = 0;
    int max_in_flight = 0;       ///< buffer_count x device_count: jobs kept on the device
    int result_buffers = 0;      ///< output blocks in this network's pool
    ExecMode exec = ExecMode::Async;
};

}  // namespace dxinfer