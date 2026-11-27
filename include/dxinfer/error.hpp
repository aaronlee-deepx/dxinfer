// Error handling for dxinfer (hybrid model):
//   * setup / misuse problems throw dxinfer::InferenceError
//     (load failure, NPU out of memory, unknown name, wrong input size, ...)
//   * a failure of ONE inference is NOT thrown: it is reported in that
//     Result's status(), so FIFO delivery and callback threads keep running.
#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace dxinfer {

enum class ErrorCode {
    Ok = 0,
    LoadFailed,         ///< a .dxnn could not be loaded (bad path, bad model, device error)
    OutOfDeviceMemory,  ///< the networks do not fit into NPU memory together
    InvalidArgument,    ///< bad option, empty network list, empty name, null input
    DuplicateName,      ///< two networks use the same name
    UnknownNetwork,     ///< name was not given at construction
    BadInputSize,       ///< input byte count != network input size
    NoCallback,         ///< CallbackInference::feed() before set_callback()
    InferenceFailed,    ///< dx-runtime rejected or failed one inference
    ShuttingDown,       ///< the object is being destroyed / queue closed
};

inline const char* to_string(ErrorCode c) noexcept
{
    switch (c) {
    case ErrorCode::Ok:                return "Ok";
    case ErrorCode::LoadFailed:        return "LoadFailed";
    case ErrorCode::OutOfDeviceMemory: return "OutOfDeviceMemory";
    case ErrorCode::InvalidArgument:   return "InvalidArgument";
    case ErrorCode::DuplicateName:     return "DuplicateName";
    case ErrorCode::UnknownNetwork:    return "UnknownNetwork";
    case ErrorCode::BadInputSize:      return "BadInputSize";
    case ErrorCode::NoCallback:        return "NoCallback";
    case ErrorCode::InferenceFailed:   return "InferenceFailed";
    case ErrorCode::ShuttingDown:      return "ShuttingDown";
    }
    return "Unknown";
}

/// Outcome of one inference, carried inside every Result.
class Status {
public:
    Status() = default;
    Status(ErrorCode c, std::string msg) : code_(c), message_(std::move(msg)) {}

    bool ok() const noexcept { return code_ == ErrorCode::Ok; }
    ErrorCode code() const noexcept { return code_; }
    const std::string& message() const noexcept { return message_; }

private:
    ErrorCode code_ = ErrorCode::Ok;
    std::string message_;
};

/// Thrown for setup errors and API misuse. what() = "[dxinfer] network 'x': Code: message".
class InferenceError : public std::runtime_error {
public:
    InferenceError(ErrorCode c, const std::string& net, const std::string& msg)
        : std::runtime_error(format(c, net, msg)), code_(c), network_(net), message_(msg) {}

    ErrorCode code() const noexcept { return code_; }
    const std::string& network() const noexcept { return network_; }  ///< "" if not network-specific
    const std::string& message() const noexcept { return message_; }  ///< message without prefix

private:
    static std::string format(ErrorCode c, const std::string& net, const std::string& msg)
    {
        std::string s = "[dxinfer] ";
        if (!net.empty()) s += "network '" + net + "': ";
        s += to_string(c);
        s += ": ";
        s += msg;
        return s;
    }

    ErrorCode code_;
    std::string network_;
    std::string message_;
};

}  // namespace dxinfer