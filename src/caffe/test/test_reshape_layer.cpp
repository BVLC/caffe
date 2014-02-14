// Copyright 2014 Sergio Guadarrama

#include <cstring>
// #include <cuda_runtime.h>

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
class ReshapeLayerTest : public ::testing::Test {
 protected:
  ReshapeLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 6, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~ReshapeLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ReshapeLayerTest, Dtypes);

TYPED_TEST(ReshapeLayerTest, TestSetup) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(1);
  layer_param.set_reshape_channels(2 * 3);
  layer_param.set_reshape_height(6);
  layer_param.set_reshape_width(5);
  ReshapeLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 2 * 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(ReshapeLayerTest, TestSetup2) {
  // Reshape like flatten
  LayerParameter layer_param;
  layer_param.set_reshape_num(2);
  layer_param.set_reshape_channels(3 * 6 * 5);
  layer_param.set_reshape_height(1);
  layer_param.set_reshape_width(1);
  ReshapeLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3 * 6 * 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(ReshapeLayerTest, TestCPU) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(1);
  layer_param.set_reshape_channels(2 * 3);
  layer_param.set_reshape_height(6);
  layer_param.set_reshape_width(5);
  ReshapeLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int c = 0; c < 2 * 3; ++c) {
    for (int h = 0; h < 6; ++h) {
      for (int w = 0; w < 5; ++w) {
        EXPECT_EQ(this->blob_top_->data_at(0, c, h, w),
          this->blob_bottom_->data_at(c / 3, c % 3, h, w));
      }
    }
  }
}

TYPED_TEST(ReshapeLayerTest, TestCPU2) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(2);
  layer_param.set_reshape_channels(3 * 6 * 5);
  layer_param.set_reshape_height(1);
  layer_param.set_reshape_width(1);
  ReshapeLayer<TypeParam> layer(layer_param);
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



TYPED_TEST(ReshapeLayerTest, TestGPU) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(1);
  layer_param.set_reshape_channels(2 * 3);
  layer_param.set_reshape_height(6);
  layer_param.set_reshape_width(5);
  ReshapeLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int c = 0; c < 2 * 3; ++c) {
    for (int h = 0; h < 6; ++h) {
      for (int w = 0; w < 5; ++w) {
        EXPECT_EQ(this->blob_top_->data_at(0, c, h, w),
          this->blob_bottom_->data_at(c / 3, c % 3, h, w));
      }
    }
  }
}

TYPED_TEST(ReshapeLayerTest, TestGPU2) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(2);
  layer_param.set_reshape_channels(3 * 6 * 5);
  layer_param.set_reshape_height(1);
  layer_param.set_reshape_width(1);
  ReshapeLayer<TypeParam> layer(layer_param);
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

TYPED_TEST(ReshapeLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(1);
  layer_param.set_reshape_channels(2 * 3);
  layer_param.set_reshape_height(6);
  layer_param.set_reshape_width(5);
  Caffe::set_mode(Caffe::CPU);
  ReshapeLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ReshapeLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  layer_param.set_reshape_num(1);
  layer_param.set_reshape_channels(2 * 3);
  layer_param.set_reshape_height(6);
  layer_param.set_reshape_width(5);
  Caffe::set_mode(Caffe::GPU);
  ReshapeLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}


}
