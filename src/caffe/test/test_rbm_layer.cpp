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
class RBMLayerTest : public ::testing::Test {
 protected:
  RBMLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~RBMLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(RBMLayerTest, Dtypes);

TYPED_TEST(RBMLayerTest, TestSetUp) {
  LayerParameter layer_param;
  RBMParameter* rbm_param = layer_param.mutable_rbm_param();
  rbm_param->set_hidden_dim(10);
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

TYPED_TEST(RBMLayerTest, TestCPU) {
  LayerParameter layer_param;
  RBMParameter* rbm_param = layer_param.mutable_rbm_param();
  Caffe::set_mode(Caffe::CPU);
  rbm_param->set_hidden_dim(10);
  rbm_param->mutable_weight_filler()->set_type("uniform");
  shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(RBMLayerTest, TestGPU) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    RBMParameter* rbm_param = layer_param.mutable_rbm_param();
    Caffe::set_mode(Caffe::GPU);
    rbm_param->set_hidden_dim(10);
    rbm_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<InnerProductLayer<TypeParam> > layer(
      new InnerProductLayer<TypeParam>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    const TypeParam* data = this->blob_top_->cpu_data();
    const int count = this->blob_top_->count();
    for (int i = 0; i < count; ++i) {
      EXPECT_GE(data[i], 1.);
    }
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(RBMLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  RBMParameter* rbm_param = layer_param.mutable_rbm_param();
  Caffe::set_mode(Caffe::CPU);
  rbm_param->set_hidden_dim(10);
  rbm_param->mutable_weight_filler()->set_type("gaussian");
  InnerProductLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(RBMLayerTest, TestGPUGradient) {
  if (sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    RBMParameter* rbm_param = layer_param.mutable_rbm_param();
    Caffe::set_mode(Caffe::GPU);
    rbm_param->set_hidden_dim(10);
    rbm_param->mutable_weight_filler()->set_type("gaussian");
    InnerProductLayer<TypeParam> layer(layer_param);
    GradientChecker<TypeParam> checker(1e-2, 1e-2);
    checker.CheckGradient(&layer, &(this->blob_bottom_vec_),
        &(this->blob_top_vec_));
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
