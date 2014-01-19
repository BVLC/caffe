// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::min;
using std::max;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class LRNLayerTest : public ::testing::Test {
 protected:
  LRNLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {};
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 7, 3, 3);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~LRNLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

template <typename Dtype>
void LRNLayerTest<Dtype>::ReferenceLRNForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  const Dtype* bottom_data = blob_bottom.cpu_data();
  Dtype* top_data = blob_top->mutable_cpu_data();
  Dtype alpha = layer_param.alpha();
  Dtype beta = layer_param.beta();
  int size = layer_param.local_size();
  for (int n = 0; n < blob_bottom.num(); ++n) {
    for (int c = 0; c < blob_bottom.channels(); ++c) {
      for (int h = 0; h < blob_bottom.height(); ++h) {
        for (int w = 0; w < blob_bottom.width(); ++w) {
          int c_start = c - (size - 1) / 2;
          int c_end = min(c_start + size, blob_bottom.channels());
          c_start = max(c_start, 0);
          Dtype scale = 1.;
          for (int i = c_start; i < c_end; ++i) {
            Dtype value = blob_bottom.data_at(n, i, h, w);
            scale += value * value * alpha / size;
          }
          *(top_data + blob_top->offset(n, c, h, w)) =
            blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
        }
      }
    }
  }
}

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(LRNLayerTest, Dtypes);

TYPED_TEST(LRNLayerTest, TestSetup) {
  LayerParameter layer_param;
  LRNLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 7);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(LRNLayerTest, TestCPUForward) {
  LayerParameter layer_param;
  LRNLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Blob<TypeParam> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(this->blob_top_->cpu_data()[i],
        top_reference.cpu_data()[i] - 1e-5);
    EXPECT_LE(this->blob_top_->cpu_data()[i],
        top_reference.cpu_data()[i] + 1e-5);
  }
}

TYPED_TEST(LRNLayerTest, TestGPUForward) {
  LayerParameter layer_param;
  LRNLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Blob<TypeParam> top_reference;
  this->ReferenceLRNForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_GE(this->blob_top_->cpu_data()[i],
        top_reference.cpu_data()[i] - 1e-5);
    EXPECT_LE(this->blob_top_->cpu_data()[i],
        top_reference.cpu_data()[i] + 1e-5);
  }
}

TYPED_TEST(LRNLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  LRNLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  //for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //  std::cout << "CPU diff " << this->blob_bottom_->cpu_diff()[i] << std::endl;
  //}
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(LRNLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  LRNLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  //for (int i = 0; i < this->blob_bottom_->count(); ++i) {
  //  std::cout << "GPU diff " << this->blob_bottom_->cpu_diff()[i] << std::endl;
  //}
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}
