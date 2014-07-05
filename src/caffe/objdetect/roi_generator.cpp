// Copyright 2014 BVLC and contributors.

#include "caffe/objdetect/roi_generator.hpp"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
SlidingWindowROIGenerator<Dtype>::SlidingWindowROIGenerator(
    const ROIGeneratorParameter& param)
  : ROIGenerator<Dtype>(param),
    num_spatial_bins_(
      this->roi_generator_param_.sliding_window_param().spatial_bin_size()),
    stride_size_ratio_(
      this->roi_generator_param_.sliding_window_param().stride_size_ratio()) {
}

template <typename Dtype>
void SlidingWindowROIGenerator<Dtype>::generate(const Blob<Dtype>& image,
                                                vector<Rect>* rois) {
  const int height = image.height();
  const int width = image.width();
  rois->clear();
  for (size_t i = 0; i < num_spatial_bins_; ++i) {
    const size_t spatial_bin =
        this->roi_generator_param_.sliding_window_param().spatial_bin(i);
    const float window_size_ratio = 1 + (spatial_bin - 1) * stride_size_ratio_;
    const int window_height = int(height / window_size_ratio);
    const int height_stride = int(window_height * stride_size_ratio_);
    const int window_width = int(width / window_size_ratio);
    const int width_stride = int(window_width * stride_size_ratio_);
    for (size_t j = 0; j < spatial_bin; ++j) {
      const int y1 = height_stride * j;
      const int y2 = std::min(height, y1 + window_height) - 1;
      for (size_t k = 0; k < spatial_bin; ++k) {
        const int x1 = width_stride * k;
        const int x2 = std::min(width, x1 + window_width) - 1;
        Rect roi(x1, y1, x2, y2);
        rois->push_back(roi);
      }
    }
  }
}


INSTANTIATE_CLASS(SlidingWindowROIGenerator);

template <typename Dtype>
ROIGenerator<Dtype>* GetROIGenerator(const ROIGeneratorParameter& param) {
  switch (param.type()) {
  case ROIGeneratorParameter_ROIGeneratorType_SLIDING_WINDOW:
    return new SlidingWindowROIGenerator<Dtype>(param);
  default:
    LOG(FATAL) << "Unknown ROIGenerator type " << param.type();
  }
}
template <>
ROIGenerator<float>* GetROIGenerator<float>(
    const ROIGeneratorParameter& param);
template <>
ROIGenerator<double>* GetROIGenerator<double>(
    const ROIGeneratorParameter& param);

}  // namespace caffe
