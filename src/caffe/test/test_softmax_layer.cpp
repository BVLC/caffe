// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test sum
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    Dtype sum = 0;
    for (int j = 0; j < this->blob_top_->channels(); ++j) {
      sum += this->blob_top_->data_at(i, j, 0, 0);
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    Dtype scale = 0;
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      scale += exp(this->blob_bottom_->data_at(i, j, 0, 0));
    }
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      EXPECT_GE(this->blob_top_->data_at(i, j, 0, 0) + 1e-4,
          exp(this->blob_bottom_->data_at(i, j, 0, 0)) / scale)
          << "debug: " << i << " " << j;
      EXPECT_LE(this->blob_top_->data_at(i, j, 0, 0) - 1e-4,
          exp(this->blob_bottom_->data_at(i, j, 0, 0)) / scale)
          << "debug: " << i << " " << j;
    }
  }
}

TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  SoftmaxLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
