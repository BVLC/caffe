// Copyright 2014 BVLC and contributors.

#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class SpatialPyramidPoolingLayerTest : public ::testing::Test {
 protected:
  SpatialPyramidPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 5, 13, 13);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SpatialPyramidPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  void TestForward() {
    const int num = 2;
    const int channels = 3;
    int offset;
    blob_bottom_->Reshape(num, channels, 5, 5);
    // Input: 2 x 3 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [31 22 15 27 33]
    //     [19 14 21 14 38]
    //     [21 32 25 12 23]
    for (int i = 0; i < 25 * num * channels; i += 25) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 5;
      blob_bottom_->mutable_cpu_data()[i +  3] = 2;
      blob_bottom_->mutable_cpu_data()[i +  4] = 3;
      blob_bottom_->mutable_cpu_data()[i +  5] = 9;
      blob_bottom_->mutable_cpu_data()[i +  6] = 4;
      blob_bottom_->mutable_cpu_data()[i +  7] = 1;
      blob_bottom_->mutable_cpu_data()[i +  8] = 4;
      blob_bottom_->mutable_cpu_data()[i +  9] = 8;
      blob_bottom_->mutable_cpu_data()[i + 10] = 31;
      blob_bottom_->mutable_cpu_data()[i + 11] = 22;
      blob_bottom_->mutable_cpu_data()[i + 12] = 15;
      blob_bottom_->mutable_cpu_data()[i + 13] = 27;
      blob_bottom_->mutable_cpu_data()[i + 14] = 33;
      blob_bottom_->mutable_cpu_data()[i + 15] = 19;
      blob_bottom_->mutable_cpu_data()[i + 16] = 14;
      blob_bottom_->mutable_cpu_data()[i + 17] = 21;
      blob_bottom_->mutable_cpu_data()[i + 18] = 14;
      blob_bottom_->mutable_cpu_data()[i + 19] = 38;
      blob_bottom_->mutable_cpu_data()[i + 20] = 21;
      blob_bottom_->mutable_cpu_data()[i + 21] = 32;
      blob_bottom_->mutable_cpu_data()[i + 22] = 25;
      blob_bottom_->mutable_cpu_data()[i + 23] = 12;
      blob_bottom_->mutable_cpu_data()[i + 24] = 23;
    }
    LayerParameter layer_param;
    SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param =
        layer_param.mutable_spatial_pyramid_pooling_param();
    spatial_pyramid_pooling_param->set_pool(
        SpatialPyramidPoolingParameter_PoolMethod_MAX);
    // kernel: 5 * 5, stride: 5 * 5
    // x1, y1, x2, y2
    // 0, 5, 0, 5
    spatial_pyramid_pooling_param->add_spatial_bin(1);
    SpatialPyramidPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels * (1 * 1));
    EXPECT_EQ(blob_top_->height(), 1);
    EXPECT_EQ(blob_top_->width(), 1);
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected: 2 x 3 channels of:
    //     [38]
    offset = 0;
    for (int i = 0; i < num; i += 1) {
      for (int pyramid_level = 0; pyramid_level < 1; ++pyramid_level) {
        for (int j = 0; j < channels; ++j) {
          if (pyramid_level == 0) {
            EXPECT_EQ(blob_top_->cpu_data()[offset], 38);
            ++offset;
          }
        }
      }
    }

    // kernel: 2.5 * 2.5, stride: 2.5 * 2.5
    // x1, y1, x2, y2
    // 0, 0, 3, 3
    // 2, 0, 5, 3
    // 0, 2, 3, 5
    // 2, 2, 5, 5
    spatial_pyramid_pooling_param->add_spatial_bin(2);
    SpatialPyramidPoolingLayer<Dtype> two_level_layer(layer_param);
    two_level_layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels * (1 * 1 + 2 * 2));
    EXPECT_EQ(blob_top_->height(), 1);
    EXPECT_EQ(blob_top_->width(), 1);
    two_level_layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected: 2 x 3 channels of:
    //     [38]
    //     [31 33]
    //     [32 38]
    offset = 0;
    for (int i = 0; i < num; i += 1) {
      for (int pyramid_level = 0; pyramid_level < 2; ++pyramid_level) {
        for (int j = 0; j < channels; ++j) {
          if (pyramid_level == 0) {
            EXPECT_EQ(blob_top_->cpu_data()[offset], 38);
            ++offset;
          } else if (pyramid_level == 1) {
            EXPECT_EQ(blob_top_->cpu_data()[offset + 0], 31);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 1], 33);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 2], 32);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 3], 38);
            offset += 4;
          }
        }
      }
    }

    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [31 22 15 27 33]
    //     [19 14 21 14 38]
    //     [21 32 25 12 23]
    // kernel: 1.666666667 * 1.666666667, stride: 1.666666667 * 1.666666667
    // x1, y1, x2, y2
    // 0, 0, 2, 2
    // 1, 0, 4, 2
    // 3, 0, 5, 2
    // 0, 1, 2, 4
    // 1, 1, 4, 4
    // 3, 1, 5, 4
    // 0, 3, 2, 5
    // 1, 3, 4, 5
    // 3, 3, 5, 5
    spatial_pyramid_pooling_param->add_spatial_bin(3);
    SpatialPyramidPoolingLayer<Dtype> three_level_layer(layer_param);
    three_level_layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels * (1 * 1 + 2 * 2 + 3 * 3));
    EXPECT_EQ(blob_top_->height(), 1);
    EXPECT_EQ(blob_top_->width(), 1);
    three_level_layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected: 2 x 3 channels of:
    //     [38]
    //     [31 33]
    //     [32 38]
    //     [9  5  8]
    //     [31 27 38]
    //     [32 32 38]
    offset = 0;
    for (int i = 0; i < num; i += 1) {
      for (int pyramid_level = 0; pyramid_level < 3; ++pyramid_level) {
        for (int j = 0; j < channels; ++j) {
          if (pyramid_level == 0) {
            EXPECT_EQ(blob_top_->cpu_data()[offset], 38);
            ++offset;
          } else if (pyramid_level == 1) {
            EXPECT_EQ(blob_top_->cpu_data()[offset + 0], 31);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 1], 33);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 2], 32);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 3], 38);
            offset += 2 * 2;
          } else if (pyramid_level == 2) {
            EXPECT_EQ(blob_top_->cpu_data()[offset + 0], 9);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 1], 5);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 2], 8);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 3], 31);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 4], 27);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 5], 38);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 6], 32);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 7], 32);
            EXPECT_EQ(blob_top_->cpu_data()[offset + 8], 38);
            offset += 3 * 3;
          }
        }
      }
    }  // for (int i = 0; i < num; i += 1) {
  }  // void TestForward() {
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(SpatialPyramidPoolingLayerTest, Dtypes);

TYPED_TEST(SpatialPyramidPoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param =
      layer_param.mutable_spatial_pyramid_pooling_param();
  spatial_pyramid_pooling_param->add_spatial_bin(1);
  spatial_pyramid_pooling_param->set_pool(
      SpatialPyramidPoolingParameter_PoolMethod_MAX);
  SpatialPyramidPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(),
            this->blob_bottom_->channels() * (1 * 1));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);

  spatial_pyramid_pooling_param->add_spatial_bin(2);
  SpatialPyramidPoolingLayer<TypeParam> two_level_layer(layer_param);
  two_level_layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(),
            this->blob_bottom_->channels() * (1 * 1 + 2 * 2));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);

  spatial_pyramid_pooling_param->add_spatial_bin(3);
  SpatialPyramidPoolingLayer<TypeParam> three_level_layer(layer_param);
  three_level_layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(),
            this->blob_bottom_->channels() * (1 * 1 + 2 * 2 + 3 * 3));
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SpatialPyramidPoolingLayerTest, TestCPUForwardMax) {
  Caffe::set_mode(Caffe::CPU);
  this->TestForward();
}

TYPED_TEST(SpatialPyramidPoolingLayerTest, TestGPUForwardMax) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForward();
}

TYPED_TEST(SpatialPyramidPoolingLayerTest, TestCPUGradientMax) {
  Caffe::set_mode(Caffe::CPU);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param =
      layer_param.mutable_spatial_pyramid_pooling_param();
  spatial_pyramid_pooling_param->set_pool(
      SpatialPyramidPoolingParameter_PoolMethod_MAX);

  for (int i = 1; i < 5; ++i) {
    spatial_pyramid_pooling_param->add_spatial_bin(i);
  }
  SpatialPyramidPoolingLayer<TypeParam> layer(layer_param);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(SpatialPyramidPoolingLayerTest, TestGPUGradientMax) {
  Caffe::set_mode(Caffe::GPU);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  vector<bool> propagate_down(this->blob_bottom_vec_.size(), true);
  LayerParameter layer_param;
  SpatialPyramidPoolingParameter* spatial_pyramid_pooling_param =
      layer_param.mutable_spatial_pyramid_pooling_param();
  spatial_pyramid_pooling_param->set_pool(
      SpatialPyramidPoolingParameter_PoolMethod_MAX);

  for (int i = 1; i < 5; ++i) {
    spatial_pyramid_pooling_param->add_spatial_bin(i);
  }
  SpatialPyramidPoolingLayer<TypeParam> layer(layer_param);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
