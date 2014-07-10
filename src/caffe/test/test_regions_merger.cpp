// Copyright 2014 BVLC and contributors.

#include "gtest/gtest.h"

#include <vector>

#include "caffe/objdetect/rect.hpp"
#include "caffe/objdetect/regions_merger.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
using std::vector;

class RegionsMergerTest : public ::testing::Test {
 protected:
  RegionsMergerTest() {}
};

TEST(RegionsMergerTest, TestNonMaximumSuppressionRegionsMerger) {
  vector<Rect> boxes;
  vector<float> confidences;
  for (int i = 0; i < 100; ++i) {
    Rect rect(i, i, i + 10, i + 10);
    boxes.push_back(rect);
    confidences.push_back(100 - i);
  }
  RegionsMergerParameter param;
  NonMaximumSuppressionRegionsMergerParameter* nms_param =
      param.mutable_nms_param();
  nms_param->set_overlap_threshold(0.5);
  NonMaximumSuppressionRegionsMerger merger(param);
  vector<int> top_boxes_indices;
  merger.merge(boxes, confidences, &top_boxes_indices);
  EXPECT_EQ(top_boxes_indices.size(), 34);
  for (size_t i = 0; i < 34; ++i) {
    EXPECT_EQ(i * 3, top_boxes_indices[i]);
  }
}

}  // namespace caffe
