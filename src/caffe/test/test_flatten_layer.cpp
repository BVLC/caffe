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
class FlattenLayerTest : public ::testing::Test {
 protected:
  FlattenLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~FlattenLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(FlattenLayerTest, Dtypes);

TYPED_TEST(FlattenLayerTest, TestSetup) {
  LayerParameter layer_param;
  FlattenLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(FlattenLayerTest, TestCPU) {
  LayerParameter layer_param;
  FlattenLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

TYPED_TEST(FlattenLayerTest, TestGPU) {
  LayerParameter layer_param;
  FlattenLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int c = 0; c < 3 * 6 * 5; ++c) {
    EXPECT_EQ(this->blob_top_->data_at(0, c, 0, 0),
        this->blob_bottom_->data_at(0, c / (6 * 5), (c / 5) % 6, c % 5));
    EXPECT_EQ(this->blob_top_->data_at(1, c, 0, 0),
        this->blob_bottom_->data_at(1, c / (6 * 5), (c / 5) % 6, c % 5));
  }
}

TYPED_TEST(FlattenLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  FlattenLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(FlattenLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  FlattenLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


}  // namespace caffe
