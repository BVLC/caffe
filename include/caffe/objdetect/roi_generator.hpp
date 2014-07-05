// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_ROI_GENERATOR_HPP_
#define CAFFE_ROI_GENERATOR_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/objdetect/rect.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using std::string;
using std::vector;

template <typename Dtype>
class ROIGenerator {
 public:
  explicit ROIGenerator(const ROIGeneratorParameter& param) :
    roi_generator_param_(param) {}
  virtual ~ROIGenerator() {}
  virtual void generate(const Blob<Dtype>& image, vector<Rect>* rois) = 0;
 protected:
  ROIGeneratorParameter roi_generator_param_;
};

template <typename Dtype>
class SlidingWindowROIGenerator : public ROIGenerator<Dtype> {
 public:
  explicit SlidingWindowROIGenerator(const ROIGeneratorParameter& param);
  virtual ~SlidingWindowROIGenerator() {}
  virtual void generate(const Blob<Dtype>& image, vector<Rect>* rois);
 protected:
  const size_t num_spatial_bins_;
  const float stride_size_ratio_;
};

template <typename Dtype>
class SelectiveSearchROIGenerator : public ROIGenerator<Dtype> {
 public:
  explicit SelectiveSearchROIGenerator(const ROIGeneratorParameter& param);
  virtual ~SelectiveSearchROIGenerator() {}
  virtual void generate(const Blob<Dtype>& image, vector<Rect>* rois);
 protected:
};

template <typename Dtype>
class BINGROIGenerator : public ROIGenerator<Dtype> {
 public:
  explicit BINGROIGenerator(const ROIGeneratorParameter& param);
  virtual ~BINGROIGenerator() {}
  virtual void generate(const Blob<Dtype>& image, vector<Rect>* rois);
 protected:
};

}  // namespace caffe

#endif  // CAFFE_ROI_GENERATOR_HPP_
