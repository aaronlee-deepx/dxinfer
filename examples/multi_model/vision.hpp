// OpenCV side of the multi-model demo: list/load images, letterbox into a model input
// buffer, draw overlays, save. Letterbox matches dx_app's makeLetterboxImage
// (BGR->RGB, pad 114, remainder to the bottom/right).
#pragma once

#include "postprocess.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace mm {

/// Image files (jpg/jpeg/png/bmp, any case) in `dir`, sorted by name. Throws std::runtime_error if unreadable.
std::vector<std::string> list_images(const std::string& dir);

/// BGR image, or throws std::runtime_error naming the file.
cv::Mat load_image(const std::string& path);

/// Letterbox `bgr` into an RGB uint8 HWC buffer of in_w x in_h x 3; fills `lb` for unletterbox().
std::vector<uint8_t> letterbox_rgb(const cv::Mat& bgr, int in_w, int in_h, Letterbox* lb);

void draw_detections(cv::Mat& img, const std::vector<Box>& boxes);  ///< COCO names
void draw_topk(cv::Mat& img, const std::vector<TopK>& top);         ///< ImageNet names

/// mkdir -p. Throws std::runtime_error on failure.
void make_dirs(const std::string& path);

/// "dir/a/b.jpg" -> "b"
std::string stem(const std::string& path);

void save_image(const std::string& path, const cv::Mat& img);  ///< throws on failure

std::string coco_name(int id);
std::string imagenet_name(int id);

}  // namespace mm