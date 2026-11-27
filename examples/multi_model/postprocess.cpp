#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace mm {

Letterbox letterbox_params(int orig_w, int orig_h, int in_w, int in_h)
{
    Letterbox lb;
    lb.orig_w = orig_w;
    lb.orig_h = orig_h;
    lb.in_w = in_w;
    lb.in_h = in_h;
    if (orig_w <= 0 || orig_h <= 0) return lb;
    lb.scale = std::min(static_cast<float>(in_w) / orig_w, static_cast<float>(in_h) / orig_h);
    lb.new_w = static_cast<int>(orig_w * lb.scale);
    lb.new_h = static_cast<int>(orig_h * lb.scale);
    lb.pad_x = (in_w - lb.new_w) / 2;
    lb.pad_y = (in_h - lb.new_h) / 2;
    return lb;
}

std::vector<Box> decode_yolo26_e2e(const float* data, int rows, int stride, float score_thr, int num_classes)
{
    std::vector<Box> out;
    if (!data || stride < 6) return out;
    for (int i = 0; i < rows; ++i) {
        const float* r = data + static_cast<size_t>(i) * stride;
        if (r[4] < score_thr) continue;
        const int cls = static_cast<int>(r[5]);
        if (cls < 0 || cls >= num_classes) continue;
        Box b;
        b.x1 = r[0];
        b.y1 = r[1];
        b.x2 = r[2];
        b.y2 = r[3];
        b.score = r[4];
        b.class_id = cls;
        out.push_back(b);
    }
    return out;
}

namespace {

float dfl_distance(const float* reg, int k, int cell, int grid)
{
    float mx = -std::numeric_limits<float>::infinity();
    for (int d = 0; d < 16; ++d) mx = std::max(mx, reg[(k * 16 + d) * grid + cell]);
    float sum = 0.f, weighted = 0.f;
    for (int d = 0; d < 16; ++d) {
        const float e = std::exp(reg[(k * 16 + d) * grid + cell] - mx);
        sum += e;
        weighted += e * static_cast<float>(d);
    }
    return weighted / sum;
}

float iou(const Box& a, const Box& b)
{
    const float l = std::max(a.x1, b.x1), t = std::max(a.y1, b.y1);
    const float r = std::min(a.x2, b.x2), btm = std::min(a.y2, b.y2);
    if (r < l || btm < t) return 0.f;
    const float inter = (r - l) * (btm - t);
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0 ? inter / uni : 0.f;
}

}  // namespace

std::vector<Box> decode_yolo26_dfl(const std::vector<Level>& levels, int input_w, int num_classes, float score_thr)
{
    std::vector<Box> out;
    for (const Level& lv : levels) {
        if (!lv.reg || !lv.cls || lv.w <= 0 || lv.h <= 0) continue;
        const int grid = lv.h * lv.w;
        const int stride = input_w / lv.w;
        for (int y = 0; y < lv.h; ++y) {
            for (int x = 0; x < lv.w; ++x) {
                const int cell = y * lv.w + x;
                int best = -1;
                float conf = score_thr;
                for (int c = 0; c < num_classes; ++c) {
                    const float v = lv.cls[c * grid + cell];
                    if (v > conf) {
                        conf = v;
                        best = c;
                    }
                }
                if (best < 0) continue;
                float d[4];
                for (int k = 0; k < 4; ++k) d[k] = dfl_distance(lv.reg, k, cell, grid);
                const float ax = x + 0.5f, ay = y + 0.5f;
                Box b;
                b.x1 = (ax - d[0]) * stride;
                b.y1 = (ay - d[1]) * stride;
                b.x2 = (ax + d[2]) * stride;
                b.y2 = (ay + d[3]) * stride;
                b.score = conf;
                b.class_id = best;
                out.push_back(b);
            }
        }
    }
    return out;
}

std::vector<Box> nms(std::vector<Box> boxes, float iou_thr)
{
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<bool> gone(boxes.size(), false);
    std::vector<Box> keep;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (gone[i]) continue;
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j)
            if (!gone[j] && iou(boxes[i], boxes[j]) > iou_thr) gone[j] = true;
    }
    return keep;
}

void unletterbox(std::vector<Box>& boxes, const Letterbox& lb)
{
    if (lb.scale <= 0.f) return;
    auto clampf = [](float v, float hi) { return std::max(0.f, std::min(v, hi)); };
    const float W = static_cast<float>(lb.orig_w), H = static_cast<float>(lb.orig_h);
    for (Box& b : boxes) {
        b.x1 = clampf((b.x1 - lb.pad_x) / lb.scale, W);
        b.y1 = clampf((b.y1 - lb.pad_y) / lb.scale, H);
        b.x2 = clampf((b.x2 - lb.pad_x) / lb.scale, W);
        b.y2 = clampf((b.y2 - lb.pad_y) / lb.scale, H);
    }
}

Elem elem_from_dxrt(int t)
{
    // dxrt::DataType: NONE_TYPE=0, FLOAT, UINT8, INT8, UINT16, INT16, INT32, INT64, UINT32, UINT64, BBOX, ...
    switch (t) {
    case 1: return Elem::F32;
    case 2: return Elem::U8;
    case 3: return Elem::I8;
    case 4: return Elem::U16;
    case 5: return Elem::I16;
    case 6: return Elem::I32;
    case 7: return Elem::I64;
    case 8: return Elem::U32;
    case 9: return Elem::U64;
    default: return Elem::Other;
    }
}

std::vector<TopK> classify(Elem type, const void* data, size_t count, int k)
{
    std::vector<TopK> out;
    if (!data || count == 0) return out;
    if (type != Elem::F32) {
        long long id = 0;
        switch (type) {
        case Elem::U8:  id = *static_cast<const uint8_t*>(data); break;
        case Elem::I8:  id = *static_cast<const int8_t*>(data); break;
        case Elem::U16: id = *static_cast<const uint16_t*>(data); break;
        case Elem::I16: id = *static_cast<const int16_t*>(data); break;
        case Elem::I32: id = *static_cast<const int32_t*>(data); break;
        case Elem::I64: id = *static_cast<const int64_t*>(data); break;
        case Elem::U32: id = *static_cast<const uint32_t*>(data); break;
        case Elem::U64: id = static_cast<long long>(*static_cast<const uint64_t*>(data)); break;
        default: return out;
        }
        TopK t;
        t.class_id = static_cast<int>(id);
        t.prob = 1.f;
        out.push_back(t);
        return out;
    }
    const float* f = static_cast<const float*>(data);
    const int n = static_cast<int>(count);
    std::vector<float> p(f, f + n);
    bool distribution = true;
    double sum = 0;
    for (float v : p) {
        if (v < -1e-4f) distribution = false;
        sum += v;
    }
    if (!(distribution && std::fabs(sum - 1.0) <= 1e-3)) {  // raw logits -> stable softmax
        const float mx = *std::max_element(p.begin(), p.end());
        double s = 0;
        for (float& v : p) {
            v = std::exp(v - mx);
            s += v;
        }
        for (float& v : p) v = static_cast<float>(v / s);
    }
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    const int kk = std::max(0, std::min(k, n));
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&p](int a, int b) { return p[a] > p[b]; });
    for (int i = 0; i < kk; ++i) {
        TopK t;
        t.class_id = idx[i];
        t.prob = p[idx[i]];
        out.push_back(t);
    }
    return out;
}

}  // namespace mm