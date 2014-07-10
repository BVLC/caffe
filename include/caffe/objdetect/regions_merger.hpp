// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_REGIONS_MERGER_HPP_
#define CAFFE_REGIONS_MERGER_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/objdetect/rect.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

using std::string;
using std::vector;

class RegionsMerger {
 public:
  explicit RegionsMerger(const RegionsMergerParameter& param) :
    regions_merger_param_(param) {}
  virtual ~RegionsMerger() {}
  virtual void merge(const vector<Rect> boxes, const vector<float> confidences,
                     vector<int>* top_boxes_indices) = 0;
 protected:
  RegionsMergerParameter regions_merger_param_;
};

class NonMaximumSuppressionRegionsMerger : public RegionsMerger {
 public:
  explicit NonMaximumSuppressionRegionsMerger(
      const RegionsMergerParameter& param);
  virtual ~NonMaximumSuppressionRegionsMerger() {}
  virtual void merge(const vector<Rect> boxes, const vector<float> confidences,
                     vector<int>* top_boxes_indices);
 protected:
  float overlap_threshold_;
};

RegionsMerger* GetRegionsMerger(const RegionsMergerParameter& param);

}  // namespace caffe

#endif  // CAFFE_REGIONS_MERGER_HPP_
