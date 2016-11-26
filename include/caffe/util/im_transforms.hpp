#ifndef IM_TRANSFORMS_HPP
#define IM_TRANSFORMS_HPP

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Generate random number given the probablities for each number.
int roll_weighted_die(const std::vector<float>& probabilities);

void UpdateBBoxByResizePolicy(const ResizeParameter& param,
                              const int old_width, const int old_height,
                              NormalizedBBox* bbox);

void InferNewSize(const ResizeParameter& resize_param,
                  const int old_width, const int old_height,
                  int* new_width, int* new_height);

#ifdef USE_OPENCV
template <typename T>
bool is_border(const cv::Mat& edge, T color);

// Auto cropping image.
template <typename T>
cv::Rect CropMask(const cv::Mat& src, T point, int padding = 2);

cv::Mat colorReduce(const cv::Mat& image, int div = 64);

void fillEdgeImage(const cv::Mat& edgesIn, cv::Mat* filledEdgesOut);

void CenterObjectAndFillBg(const cv::Mat& in_img, const bool fill_bg,
                           cv::Mat* out_img);

cv::Mat AspectKeepingResizeAndPad(const cv::Mat& in_img,
                                  const int new_width, const int new_height,
                                  const int pad_type = cv::BORDER_CONSTANT,
                                  const cv::Scalar pad = cv::Scalar(0, 0, 0),
                                  const int interp_mode = cv::INTER_LINEAR);

cv::Mat AspectKeepingResizeBySmall(const cv::Mat& in_img,
                                   const int new_width, const int new_height,
                                   const int interp_mode = cv::INTER_LINEAR);

void constantNoise(const int n, const vector<uchar>& val, cv::Mat* image);

cv::Mat ApplyResize(const cv::Mat& in_img, const ResizeParameter& param);

cv::Mat ApplyNoise(const cv::Mat& in_img, const NoiseParameter& param);


void RandomBrightness(const cv::Mat& in_img, cv::Mat* out_img,
    const float brightness_prob, const float brightness_delta);

void AdjustBrightness(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);

void RandomContrast(const cv::Mat& in_img, cv::Mat* out_img,
    const float contrast_prob, const float lower, const float upper);

void AdjustContrast(const cv::Mat& in_img, const float delta,
                    cv::Mat* out_img);

void RandomSaturation(const cv::Mat& in_img, cv::Mat* out_img,
    const float saturation_prob, const float lower, const float upper);

void AdjustSaturation(const cv::Mat& in_img, const float delta,
                      cv::Mat* out_img);

void RandomHue(const cv::Mat& in_img, cv::Mat* out_img,
               const float hue_prob, const float hue_delta);

void AdjustHue(const cv::Mat& in_img, const float delta, cv::Mat* out_img);

void RandomOrderChannels(const cv::Mat& in_img, cv::Mat* out_img,
                         const float random_order_prob);

cv::Mat ApplyDistort(const cv::Mat& in_img, const DistortionParameter& param);
#endif  // USE_OPENCV

}  // namespace caffe

#endif  // IM_TRANSFORMS_HPP
