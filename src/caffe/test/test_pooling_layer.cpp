// Copyright 2013 Yangqing Jia

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
class PoolingLayerTest : public ::testing::Test {
 protected:
  PoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 3, 6, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PoolingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(PoolingLayerTest, Dtypes);

TYPED_TEST(PoolingLayerTest, TestSetup) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

/*
TYPED_TEST(PoolingLayerTest, PrintGPUBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(PoolingLayerTest, TestCPUGradientMax) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::CPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(PoolingLayerTest, TestGPUGradientMax) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_MAX);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestCPUGradientAve) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::CPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


TYPED_TEST(PoolingLayerTest, TestGPUGradientAve) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(LayerParameter_PoolMethod_AVE);
  Caffe::set_mode(Caffe::GPU);
  PoolingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}


}  // namespace caffe
