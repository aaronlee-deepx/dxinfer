// Model factories for the multi-model demo, in the shape of dx_app's IFactory
// (createPreprocessor / createPostprocessor / createVisualizer / getModelName / getTaskType).
// The model logic is ported from dx_app's yolo26-n_640x640 and resnet factories; it works
// on dxinfer Results because dx_app's processors are tied to dxrt_api.h tensors (see README).
#pragma once

#include "postprocess.hpp"
#include "vision.hpp"

#include <dxinfer/result.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mm {

/// A preprocessed input ready for feed(), plus what post-processing needs to undo the letterbox.
struct Prepared {
    std::vector<uint8_t> input;
    Letterbox lb;
};

class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;
    virtual Prepared process(const cv::Mat& bgr) const = 0;
};

template <typename R>
class IPostprocessor {
public:
    virtual ~IPostprocessor() = default;
    virtual R process(const dxinfer::Result& r, const Letterbox& lb) = 0;
};

template <typename R>
class IVisualizer {
public:
    virtual ~IVisualizer() = default;
    virtual void draw(cv::Mat& img, const R& result) const = 0;
};

template <typename R>
class IFactory {
public:
    virtual ~IFactory() = default;
    virtual std::unique_ptr<IPreprocessor> createPreprocessor(int input_width, int input_height) = 0;
    virtual std::unique_ptr<IPostprocessor<R>> createPostprocessor(int input_width, int input_height) = 0;
    virtual std::unique_ptr<IVisualizer<R>> createVisualizer() = 0;
    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;
};

using Detections = std::vector<Box>;
using TopKList = std::vector<TopK>;

/// Letterbox + BGR->RGB + pad 114 (dx_app DetectionPreprocessor; dx_app uses it for resnet too).
class LetterboxPreprocessor : public IPreprocessor {
public:
    LetterboxPreprocessor(int w, int h) : w_(w), h_(h) {}
    Prepared process(const cv::Mat& bgr) const override
    {
        Prepared p;
        p.input = letterbox_rgb(bgr, w_, h_, &p.lb);
        return p;
    }

private:
    int w_, h_;
};

inline int64_t element_count(const std::vector<int64_t>& shape)
{
    if (shape.empty()) return 0;
    int64_t n = 1;
    for (int64_t d : shape) {
        if (d <= 0) return 0;
        n *= d;
    }
    return n;
}

// ------------------------------------------------------------------ yolo26-n (object detection)

class Yolo26Postprocessor : public IPostprocessor<Detections> {
public:
    Yolo26Postprocessor(int in_w, float score, float iou, int num_classes)
        : in_w_(in_w), score_(score), iou_(iou), classes_(num_classes) {}

    Detections process(const dxinfer::Result& r, const Letterbox& lb) override
    {
        Detections boxes;
        const auto& ts = r.tensors();
        // ONNX Runtime on: end-to-end output [1, 300, 6] (x1, y1, x2, y2, score, class)
        for (const auto& t : ts) {
            if (t.shape.size() == 3 && t.shape[2] >= 6 && elem_from_dxrt(t.dtype) == Elem::F32) {
                boxes = decode_yolo26_e2e(t.as<float>(), static_cast<int>(t.shape[1]), static_cast<int>(t.shape[2]),
                                          score_, classes_);
                return finish(boxes, lb);
            }
        }
        // ORT off: the NPU's raw heads, reg [1,64,H,W] + cls [1,C,H,W] per stride
        std::vector<const dxinfer::TensorView*> reg, cls;
        for (const auto& t : ts) {
            if (t.shape.size() != 4 || elem_from_dxrt(t.dtype) != Elem::F32) continue;
            if (t.shape[1] == 64) reg.push_back(&t);
            else if (t.shape[1] == classes_) cls.push_back(&t);
        }
        if (reg.size() == 3 && cls.size() == 3) {
            auto by_size = [](const dxinfer::TensorView* a, const dxinfer::TensorView* b) { return a->shape[2] > b->shape[2]; };
            std::sort(reg.begin(), reg.end(), by_size);
            std::sort(cls.begin(), cls.end(), by_size);
            std::vector<Level> levels;
            for (int i = 0; i < 3; ++i)
                levels.push_back(Level{reg[i]->as<float>(), cls[i]->as<float>(), static_cast<int>(cls[i]->shape[2]),
                                       static_cast<int>(cls[i]->shape[3])});
            boxes = decode_yolo26_dfl(levels, in_w_, classes_, score_);
        }
        return finish(boxes, lb);
    }

private:
    Detections finish(Detections& boxes, const Letterbox& lb) const
    {
        Detections kept = nms(boxes, iou_);
        unletterbox(kept, lb);
        return kept;
    }
    int in_w_;
    float score_, iou_;
    int classes_;
};

class DetectionVisualizer : public IVisualizer<Detections> {
public:
    void draw(cv::Mat& img, const Detections& d) const override { draw_detections(img, d); }
};

class Yolo26nFactory : public IFactory<Detections> {
public:
    explicit Yolo26nFactory(float score = 0.3f, float iou = 0.45f) : score_(score), iou_(iou) {}
    std::unique_ptr<IPreprocessor> createPreprocessor(int w, int h) override
    {
        return std::unique_ptr<IPreprocessor>(new LetterboxPreprocessor(w, h));
    }
    std::unique_ptr<IPostprocessor<Detections>> createPostprocessor(int w, int /*h*/) override
    {
        return std::unique_ptr<IPostprocessor<Detections>>(new Yolo26Postprocessor(w, score_, iou_, 80));
    }
    std::unique_ptr<IVisualizer<Detections>> createVisualizer() override
    {
        return std::unique_ptr<IVisualizer<Detections>>(new DetectionVisualizer());
    }
    std::string getModelName() const override { return "yolo26-n_640x640"; }
    std::string getTaskType() const override { return "object_detection"; }

private:
    float score_, iou_;
};

// ------------------------------------------------------------------ resnet18 (image classification)

class TopKPostprocessor : public IPostprocessor<TopKList> {
public:
    explicit TopKPostprocessor(int k) : k_(k) {}
    TopKList process(const dxinfer::Result& r, const Letterbox& /*lb*/) override
    {
        if (r.tensor_count() == 0) return {};
        const dxinfer::TensorView& t = r.tensor(0);
        return classify(elem_from_dxrt(t.dtype), t.data, static_cast<size_t>(element_count(t.shape)), k_);
    }

private:
    int k_;
};

class ClassificationVisualizer : public IVisualizer<TopKList> {
public:
    void draw(cv::Mat& img, const TopKList& t) const override { draw_topk(img, t); }
};

class Resnet18Factory : public IFactory<TopKList> {
public:
    explicit Resnet18Factory(int top_k = 5) : k_(top_k) {}
    std::unique_ptr<IPreprocessor> createPreprocessor(int w, int h) override
    {
        return std::unique_ptr<IPreprocessor>(new LetterboxPreprocessor(w, h));
    }
    std::unique_ptr<IPostprocessor<TopKList>> createPostprocessor(int, int) override
    {
        return std::unique_ptr<IPostprocessor<TopKList>>(new TopKPostprocessor(k_));
    }
    std::unique_ptr<IVisualizer<TopKList>> createVisualizer() override
    {
        return std::unique_ptr<IVisualizer<TopKList>>(new ClassificationVisualizer());
    }
    std::string getModelName() const override { return "resnet18_224x224"; }
    std::string getTaskType() const override { return "image_classification"; }

private:
    int k_;
};

}  // namespace mm