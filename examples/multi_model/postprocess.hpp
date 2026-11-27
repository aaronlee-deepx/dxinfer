// Post-processing for the multi-model demo. No OpenCV, no dx-runtime: host unit-tested.
// Ported from dx_app (src/cpp_example/common/utility/preprocessing.hpp,
// src/postprocess/yolov26/yolov26_postprocess.cpp,
// src/cpp_example/common/processors/classification_postprocessor.hpp), rewritten to work on the
// raw output pointers dxinfer hands back instead of dx_app's dxrt::TensorPtrs.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mm {

/// Letterbox geometry: aspect-preserving resize to fit, centred padding (remainder bottom/right).
struct Letterbox {
    float scale = 1.f;
    int new_w = 0, new_h = 0;  ///< image size inside the model input
    int pad_x = 0, pad_y = 0;  ///< left / top padding
    int orig_w = 0, orig_h = 0;
    int in_w = 0, in_h = 0;
};
Letterbox letterbox_params(int orig_w, int orig_h, int in_w, int in_h);

/// One detection in pixel coordinates (model input space until unletterbox()).
struct Box {
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    float score = 0;
    int class_id = 0;
};

/// YOLOv26 with ONNX Runtime (end-to-end output [1, N, stride>=6]: x1,y1,x2,y2,score,class).
std::vector<Box> decode_yolo26_e2e(const float* data, int rows, int stride, float score_thr, int num_classes);

/// One raw NPU level for ort=off: reg [1,64,H,W] (4 sides x 16 DFL bins), cls [1,C,H,W] (sigmoid scores).
struct Level {
    const float* reg;
    const float* cls;
    int h, w;
};
std::vector<Box> decode_yolo26_dfl(const std::vector<Level>& levels, int input_w, int num_classes, float score_thr);

/// Greedy NMS by score, class-agnostic like dx_app's YOLOv26PostProcess::apply_nms.
std::vector<Box> nms(std::vector<Box> boxes, float iou_thr);

/// Map boxes from model-input pixels back to the original image, clamped to its size.
void unletterbox(std::vector<Box>& boxes, const Letterbox& lb);

/// Element type of a raw output tensor.
enum class Elem { F32, U8, I8, U16, I16, I32, I64, U32, U64, Other };
Elem elem_from_dxrt(int dxrt_data_type);  ///< dxrt::DataType value (dxinfer TensorView::dtype)

struct TopK {
    int class_id = 0;
    float prob = 0;
};
/// Float scores: top-k probabilities (softmax unless the output already is a distribution).
/// Integer output: the first value is the class id (prob 1). Unsupported type: empty.
std::vector<TopK> classify(Elem type, const void* data, size_t count, int k);

}  // namespace mm