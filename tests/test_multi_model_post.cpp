// Host tests for the multi-model demo's post-processing (no OpenCV, no NPU).
// Logic ported from dx_app: letterbox (utility/preprocessing.hpp), YOLOv26 decode + NMS
// (postprocess/yolov26), classification top-k (processors/classification_postprocessor.hpp).
#include "postprocess.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace mm;

static bool near(float a, float b, float eps = 1e-3f) { return std::fabs(a - b) <= eps; }

static void letterbox_matches_dx_app()
{
    const Letterbox lb = letterbox_params(1280, 720, 640, 640);  // scale 0.5 -> 640x360, pad 140 top
    CHECK(near(lb.scale, 0.5f));
    CHECK(lb.new_w == 640 && lb.new_h == 360);
    CHECK(lb.pad_x == 0 && lb.pad_y == 140);
    const Letterbox sq = letterbox_params(500, 375, 224, 224);
    CHECK(sq.new_w == 224 && sq.pad_x == 0 && sq.pad_y == (224 - sq.new_h) / 2);
}

static void e2e_rows_threshold_and_bad_class()
{
    const float d[] = {10, 150, 110, 250, 0.9f, 0,     // kept
                       0, 0, 10, 10, 0.1f, 1,          // below the score threshold
                       20, 160, 60, 200, 0.8f, 99,     // class id out of range
                       300, 300, 400, 400, 0.5f, 79};  // kept
    const std::vector<Box> b = decode_yolo26_e2e(d, 4, 6, 0.3f, 80);
    CHECK(b.size() == 2);
    CHECK(b.size() == 2 && b[0].class_id == 0 && near(b[0].score, 0.9f));
    CHECK(b.size() == 2 && b[1].class_id == 79);
}

static void unletterbox_maps_to_original_and_clamps()
{
    const Letterbox lb = letterbox_params(1280, 720, 640, 640);
    std::vector<Box> b{{10, 150, 110, 250, 0.9f, 0}};
    unletterbox(b, lb);
    CHECK(near(b[0].x1, 20) && near(b[0].y1, 20) && near(b[0].x2, 220) && near(b[0].y2, 220));
    std::vector<Box> c{{-5, 100, 700, 700, 0.9f, 0}};
    unletterbox(c, lb);
    CHECK(near(c[0].x1, 0) && near(c[0].y1, 0) && near(c[0].x2, 1280) && near(c[0].y2, 720));
}

static void nms_suppresses_overlaps()
{
    std::vector<Box> b{{0, 0, 100, 100, 0.6f, 0}, {5, 5, 105, 105, 0.9f, 0}, {200, 200, 300, 300, 0.7f, 1}};
    const std::vector<Box> r = nms(b, 0.45f);
    CHECK(r.size() == 2);
    CHECK(r.size() == 2 && near(r[0].score, 0.9f) && near(r[1].score, 0.7f));
}

static void dfl_grid_decode()
{
    // one level, 2x2 grid, 64 px input -> stride 32, 3 classes
    const int H = 2, W = 2, G = H * W, s = 2;  // cell h=1, w=0
    std::vector<float> reg(64 * G, 0.f), cls(3 * G, 0.f);
    cls[2 * G + s] = 0.8f;
    for (int k = 0; k < 4; ++k) reg[(k * 16 + 1) * G + s] = 20.f;  // every side: distance ~1 cell
    const std::vector<Level> lv{{reg.data(), cls.data(), H, W}};
    const std::vector<Box> b = decode_yolo26_dfl(lv, 64, 3, 0.3f);
    CHECK(b.size() == 1);
    CHECK(b.size() == 1 && b[0].class_id == 2 && near(b[0].score, 0.8f));
    CHECK(b.size() == 1 && near(b[0].x1, -16, 0.01f) && near(b[0].y1, 16, 0.01f) && near(b[0].x2, 48, 0.01f) &&
          near(b[0].y2, 80, 0.01f));
}

static void topk_softmax_from_logits()
{
    const float d[] = {1, 3, 2, 0};
    const std::vector<TopK> t = classify(Elem::F32, d, 4, 2);
    CHECK(t.size() == 2);
    CHECK(t.size() == 2 && t[0].class_id == 1 && t[1].class_id == 2);
    const float denom = std::exp(0.f) + std::exp(1.f) + std::exp(2.f) + std::exp(3.f);
    CHECK(t.size() == 2 && near(t[0].prob, std::exp(3.f) / denom));
    CHECK(classify(Elem::F32, d, 4, 10).size() == 4);  // k clamped to the class count
}

static void topk_keeps_an_existing_distribution()
{
    const float d[] = {0.1f, 0.7f, 0.2f};
    const std::vector<TopK> t = classify(Elem::F32, d, 3, 3);
    CHECK(t.size() == 3 && t[0].class_id == 1 && near(t[0].prob, 0.7f));  // no double softmax
}

static void integer_output_is_the_class_id()
{
    const int64_t v = 207;
    std::vector<TopK> t = classify(Elem::I64, &v, 1, 5);
    CHECK(t.size() == 1 && t[0].class_id == 207 && near(t[0].prob, 1.f));
    const uint8_t u = 3;
    t = classify(Elem::U8, &u, 1, 5);
    CHECK(t.size() == 1 && t[0].class_id == 3);
    CHECK(classify(Elem::Other, &u, 1, 5).empty());
}

static void dxrt_dtype_mapping()
{
    CHECK(elem_from_dxrt(1) == Elem::F32);
    CHECK(elem_from_dxrt(2) == Elem::U8);
    CHECK(elem_from_dxrt(6) == Elem::I32);
    CHECK(elem_from_dxrt(7) == Elem::I64);
    CHECK(elem_from_dxrt(10) == Elem::Other);  // BBOX
}

int main()
{
    RUN(letterbox_matches_dx_app);
    RUN(e2e_rows_threshold_and_bad_class);
    RUN(unletterbox_maps_to_original_and_clamps);
    RUN(nms_suppresses_overlaps);
    RUN(dfl_grid_decode);
    RUN(topk_softmax_from_logits);
    RUN(topk_keeps_an_existing_distribution);
    RUN(integer_output_is_the_class_id);
    RUN(dxrt_dtype_mapping);
    return report("test_multi_model_post");
}