// Copyright 2014 BVLC and contributors.

#include "caffe/objdetect/regions_merger.hpp"

#include <algorithm>
#include <utility>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

NonMaximumSuppressionRegionsMerger::NonMaximumSuppressionRegionsMerger(
    const RegionsMergerParameter& param) : RegionsMerger(param),
        overlap_threshold_(
            this->regions_merger_param_.nms_param().overlap_threshold()) {
}

template<typename Dtype>
bool Dtype_int_pair_greater(std::pair<Dtype, int> a,
                            std::pair<Dtype, int> b) {
  return a.first > b.first || (a.first == b.first && a.second > b.second);
}

// https://github.com/quantombone/exemplarsvm/blob/master/internal/esvm_nms.m
void NonMaximumSuppressionRegionsMerger::merge(
    const vector<Rect> boxes, const vector<float> confidences,
    vector<int>* top_boxes_indices) {
  top_boxes_indices->clear();
  if (boxes.size() <= 0) {
    return;
  }

  vector<float> areas;
  for (size_t i = 0; i < boxes.size(); ++i) {
    areas.push_back(boxes[i].area());
  }

  vector<std::pair<float, int> > value_and_indices;
  for (size_t i = 0; i < confidences.size(); ++i) {
    value_and_indices.push_back(std::make_pair(confidences[i], i));
  }
  std::sort(value_and_indices.begin(), value_and_indices.end(),
            Dtype_int_pair_greater<float>);

  float area;
  vector<bool> is_candidate(confidences.size(), true);
  for (size_t i = 0; i < value_and_indices.size(); ++i) {
    if (!is_candidate[i]) {
      continue;
    }
    top_boxes_indices->push_back(value_and_indices[i].second);
    for (size_t j = i + 1; j < value_and_indices.size(); ++j) {
      if (is_candidate[j]) {
        area = boxes[value_and_indices[i].second].intersect(
            boxes[value_and_indices[j].second]).area();
        if (area / areas[value_and_indices[j].second] > overlap_threshold_) {
          is_candidate[j] = false;
        }
      }
    }
  }
}

RegionsMerger* GetRegionsMerger(const RegionsMergerParameter& param) {
  switch (param.type()) {
  case RegionsMergerParameter_RegionsMergerType_NON_MAXIMUM_SUPPRESION:
    return new NonMaximumSuppressionRegionsMerger(param);
  default:
    LOG(FATAL) << "Unknown RegionsMerger type " << param.type();
  }
}

}  // namespace caffe
