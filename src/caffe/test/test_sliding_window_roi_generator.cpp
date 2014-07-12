// Copyright 2014 BVLC and contributors.

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/objdetect/roi_generator.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class SlidingWindowROIGeneratorTest : public ::testing::Test {};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SlidingWindowROIGeneratorTest, Dtypes);

TYPED_TEST(SlidingWindowROIGeneratorTest, TestSpatialBin) {
  Blob<TypeParam> image(1, 3, 100, 200);
  vector<Rect> rois;
  for (int k = 0; k < 10; ++k) {
    ROIGeneratorParameter param;
    SlidingWindowROIGeneratorParameter* sliding_window_param =
        param.mutable_sliding_window_param();
    sliding_window_param->set_stride_size_ratio(k * 0.1);
    size_t counter = 0;
    for (int i = 0; i < 10; ++i) {
      sliding_window_param->add_spatial_bin(i + 1);
      SlidingWindowROIGenerator<TypeParam> generator(param);
      generator.generate(image, &rois);
      counter += (i + 1) * (i + 1);
      EXPECT_EQ(rois.size(), counter);
    }
  }
}

TYPED_TEST(SlidingWindowROIGeneratorTest, TestROI) {
  const int height = 224;
  const int width = 224;
  const Blob<TypeParam> image(1, 3, height, width);
  vector<Rect> rois;
  ROIGeneratorParameter roi_generator_param;
  SlidingWindowROIGeneratorParameter* sliding_window_param =
      roi_generator_param.mutable_sliding_window_param();
  sliding_window_param->set_stride_size_ratio(0.5);

  sliding_window_param->add_spatial_bin(1);
  SlidingWindowROIGenerator<TypeParam> generator_single_level(
      roi_generator_param);
  generator_single_level.generate(image, &rois);
  EXPECT_EQ(rois.size(), 1);
  Rect roi_full(0, 0, width, height);
  EXPECT_EQ(rois[0], roi_full);

  sliding_window_param->add_spatial_bin(3);
  SlidingWindowROIGenerator<TypeParam> generator_two_levels(
      roi_generator_param);
  generator_two_levels.generate(image, &rois);
  EXPECT_EQ(rois.size(), 1 + 3 * 3);
  EXPECT_EQ(rois[0], roi_full);
  int level2_x1[] = {0, 56, 112};
  int level2_y1[] = {0, 56, 112};
  int level2_x2[] = {112, 168, 224};
  int level2_y2[] = {112, 168, 224};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Rect roi(level2_x1[j], level2_y1[i], level2_x2[j], level2_y2[i]);
      EXPECT_EQ(rois[1 + i * 3 + j], roi);
    }
  }

  sliding_window_param->add_spatial_bin(6);
  SlidingWindowROIGenerator<TypeParam> generator_three_levels(
      roi_generator_param);
  generator_three_levels.generate(image, &rois);
  EXPECT_EQ(rois.size(), 1 + 3 * 3 + 6 * 6);
  EXPECT_EQ(rois[0], roi_full);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Rect roi(level2_x1[j], level2_y1[i], level2_x2[j], level2_y2[i]);
      EXPECT_EQ(rois[1 + i * 3 + j], roi);
    }
  }
  int level3_x1[] = {0, 32, 64, 96, 128, 160};
  int level3_y1[] = {0, 32, 64, 96, 128, 160};
  int level3_x2[] = {64, 96, 128, 160, 192, 224};
  int level3_y2[] = {64, 96, 128, 160, 192, 224};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      Rect roi(level3_x1[j], level3_y1[i], level3_x2[j], level3_y2[i]);
      EXPECT_EQ(rois[1 + 3 * 3 + i * 6 + j], roi);
    }
  }
}

}  // namespace caffe
