#include "vision.hpp"

#include "labels.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>

namespace mm {

namespace {

std::string lower_ext(const std::string& p)
{
    const size_t dot = p.find_last_of('.');
    if (dot == std::string::npos) return "";
    std::string e = p.substr(dot + 1);
    std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return e;
}

cv::Scalar class_color(int id)  // stable, distinct colours per class
{
    static const cv::Scalar palette[] = {{56, 56, 255}, {151, 157, 255}, {31, 112, 255}, {29, 178, 255},
                                         {49, 210, 207}, {10, 249, 72},  {23, 204, 146}, {134, 219, 61},
                                         {52, 147, 26},  {187, 212, 0},  {168, 153, 44}, {255, 194, 0},
                                         {147, 69, 52},  {255, 115, 100}, {236, 24, 0},  {255, 56, 132}};
    return palette[(id < 0 ? -id : id) % 16];
}

void label_box(cv::Mat& img, const std::string& text, cv::Point at, const cv::Scalar& bg, double scale)
{
    int base = 0;
    const int thick = std::max(1, static_cast<int>(scale * 2));
    const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thick, &base);
    const int y = std::max(at.y, ts.height + 6);
    cv::rectangle(img, cv::Point(at.x, y - ts.height - 6), cv::Point(at.x + ts.width + 6, y), bg, cv::FILLED);
    cv::putText(img, text, cv::Point(at.x + 3, y - 4), cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
                thick, cv::LINE_AA);
}

}  // namespace

std::vector<std::string> list_images(const std::string& dir)
{
    struct stat st;
    if (::stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        throw std::runtime_error("image directory not found: " + dir);
    std::vector<cv::String> all;
    cv::glob(dir + "/*", all, false);
    std::vector<std::string> out;
    for (const cv::String& f : all) {
        const std::string e = lower_ext(f);
        if (e == "jpg" || e == "jpeg" || e == "png" || e == "bmp") out.push_back(f);
    }
    std::sort(out.begin(), out.end());
    return out;
}

cv::Mat load_image(const std::string& path)
{
    cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
    if (m.empty()) throw std::runtime_error("cannot read image: " + path);
    return m;
}

std::vector<uint8_t> letterbox_rgb(const cv::Mat& bgr, int in_w, int in_h, Letterbox* lb)
{
    *lb = letterbox_params(bgr.cols, bgr.rows, in_w, in_h);
    cv::Mat resized, rgb, padded;
    cv::resize(bgr, resized, cv::Size(lb->new_w, lb->new_h));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    cv::copyMakeBorder(rgb, padded, lb->pad_y, in_h - lb->new_h - lb->pad_y, lb->pad_x, in_w - lb->new_w - lb->pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (!padded.isContinuous()) padded = padded.clone();
    return std::vector<uint8_t>(padded.data, padded.data + padded.total() * padded.elemSize());
}

void draw_detections(cv::Mat& img, const std::vector<Box>& boxes)
{
    const double scale = std::max(0.45, std::min(img.cols, img.rows) / 900.0);
    const int thick = std::max(2, static_cast<int>(scale * 3));
    for (const Box& b : boxes) {
        const cv::Scalar c = class_color(b.class_id);
        cv::rectangle(img, cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1)),
                      cv::Point(static_cast<int>(b.x2), static_cast<int>(b.y2)), c, thick);
        char text[96];
        std::snprintf(text, sizeof(text), "%s %.2f", coco_name(b.class_id).c_str(), b.score);
        label_box(img, text, cv::Point(static_cast<int>(b.x1), static_cast<int>(b.y1)), c, scale);
    }
}

void draw_topk(cv::Mat& img, const std::vector<TopK>& top)
{
    const double scale = std::max(0.5, std::min(img.cols, img.rows) / 700.0);
    int y = 0;
    for (size_t i = 0; i < top.size(); ++i) {
        char text[128];
        std::snprintf(text, sizeof(text), "%zu. %s %.1f%%", i + 1, imagenet_name(top[i].class_id).c_str(),
                      top[i].prob * 100.0);
        int base = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &base);
        y += ts.height + 10;
        label_box(img, text, cv::Point(6, y), i == 0 ? cv::Scalar(40, 140, 30) : cv::Scalar(60, 60, 60), scale);
    }
}

void make_dirs(const std::string& path)
{
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && ::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST)
                throw std::runtime_error("cannot create directory " + cur + ": " + std::strerror(errno));
        }
        if (i < path.size()) cur += path[i];
    }
}

std::string stem(const std::string& path)
{
    const size_t slash = path.find_last_of('/');
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? name : name.substr(0, dot);
}

void save_image(const std::string& path, const cv::Mat& img)
{
    if (!cv::imwrite(path, img)) throw std::runtime_error("cannot write image: " + path);
}

std::string coco_name(int id)
{
    const auto& l = dxapp::getCOCO80Labels();
    return id >= 0 && id < static_cast<int>(l.size()) ? l[id] : "class " + std::to_string(id);
}

std::string imagenet_name(int id) { return dxapp::getImageNetClassName(id); }

}  // namespace mm