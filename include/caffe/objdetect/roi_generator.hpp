// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_ROI_GENERATOR_HPP_
#define CAFFE_ROI_GENERATOR_HPP_

#include <sstream>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using std::string;
using std::ostringstream;
using std::vector;

class Rect {
 public:
  Rect(const int x1, const int y1, const int x2, const int y2) : x1_(x1),
    y1_(y1), x2_(x2), y2_(y2) {
  }

  Rect(const Rect& other) : x1_(other.x1()),
    y1_(other.y1()), x2_(other.x2()), y2_(other.y2()) {
  }

  Rect& operator = (const Rect& other) {
    x1_ = other.x1();
    y1_ = other.y1();
    x2_ = other.x2();
    y2_ = other.y2();
    return *this;
  }

  bool operator == (const Rect& other) const {
    return x1_ == other.x1() && y1_ == other.y1() && x2_ == other.x2() &&
        y2_ == other.y2();
  }

  inline int x1() { return x1_; }
  inline int y1() { return y1_; }
  inline int x2() { return x2_; }
  inline int y2() { return y2_; }
  inline const int x1() const { return x1_; }
  inline const int y1() const { return y1_; }
  inline const int x2() const { return x2_; }
  inline const int y2() const { return y2_; }

 protected:
  // All are inclusive
  int x1_;  // left upper corner horizontal
  int y1_;  // left upper corner vertical
  int x2_;  // right bottom corner horizontal
  int y2_;  // right bottom corner vertical
};

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
