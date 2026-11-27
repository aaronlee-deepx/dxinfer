// Read a compiled model's task graph (which parts run on NPU / CPU) straight
// from the .dxnn file, before dx-runtime loads it, and decide whether ONNX
// Runtime must run. dx-runtime itself does not expose this graph publicly.
//
// All .dxnn format knowledge lives in src/model_graph.cpp. When a file cannot
// be read (e.g. a new format version), the graph is GraphKind::Unknown with a
// reason in ModelGraph::note, resolve_ort() falls back to ORT on and returns a
// WARNING so the format change is noticed instead of silently ignored.
#pragma once

#include "dxinfer/types.hpp"

#include <string>
#include <vector>

namespace dxinfer {

/// .dxnn file format versions this reader understands.
constexpr int kGraphReaderMinFormat = 7;
constexpr int kGraphReaderMaxFormat = 9;

/// Reads only the file header and the graph info (never the weights). Never throws.
ModelGraph read_model_graph(const std::string& path) noexcept;

/// "npu_0[NPU] -> cpu_0[CPU]"
std::string describe(const ModelGraph& g);

struct OrtDecision {
    bool use_ort = false;               ///< value for dxrt::InferenceOption::useORT
    bool required = false;              ///< the model cannot run correctly without ORT
    std::vector<std::string> messages;  ///< hints / warnings for stderr, one line each
};

/// Decide useORT for one network. Throws InferenceError(InvalidArgument) for
/// OrtMode::Off on a model with 2+ NPU tasks (dx-runtime would run only the first one).
OrtDecision resolve_ort(const std::string& network, const std::string& path, OrtMode mode, const ModelGraph& g);

/// Compare the decision with what dx-runtime actually loaded (GetTaskOrder().size(),
/// IsOrtConfigured()). Throws InferenceError(LoadFailed) when the model needs ORT but
/// dx-runtime ran it without; returns warnings otherwise (empty = all consistent).
std::vector<std::string> check_loaded(const std::string& network, const OrtDecision& d, const ModelGraph& g,
                                      int loaded_tasks, bool ort_configured);

}  // namespace dxinfer