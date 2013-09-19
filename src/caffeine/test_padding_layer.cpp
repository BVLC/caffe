#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/blob.hpp"
#include "caffeine/common.hpp"
#include "caffeine/filler.hpp"
#include "caffeine/vision_layers.hpp"
#include "caffeine/test/test_gradient_check_util.hpp"


namespace caffeine {

extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;
  
template <typename Dtype>
class PaddingLayerTest : public ::testing::Test {
 protected:
  PaddingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~PaddingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(PaddingLayerTest, Dtypes);

TYPED_TEST(PaddingLayerTest, TestCPU) {
  LayerParameter layer_param;
  layer_param.set_pad(1);
  Caffeine::set_mode(Caffeine::CPU);
  PaddingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 7);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 5; ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_->data_at(n, c, h + 1, w + 1));
        }
      }
    }
  }
}

TYPED_TEST(PaddingLayerTest, TestCPUGrad) {
  LayerParameter layer_param;
  layer_param.set_pad(1);
  Caffeine::set_mode(Caffeine::CPU);
  PaddingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(PaddingLayerTest, TestGPU) {
  LayerParameter layer_param;
  layer_param.set_pad(1);
  Caffeine::set_mode(Caffeine::GPU);
  PaddingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 6);
  EXPECT_EQ(this->blob_top_->width(), 7);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 5; ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_->data_at(n, c, h + 1, w + 1));
        }
      }
    }
  }
}

TYPED_TEST(PaddingLayerTest, TestGPUGrad) {
  LayerParameter layer_param;
  layer_param.set_pad(1);
  Caffeine::set_mode(Caffeine::GPU);
  PaddingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradient(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}
