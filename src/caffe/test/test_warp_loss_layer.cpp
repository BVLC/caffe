// Copyright 2014 kloudkl@github

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

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
class WARPLossLayerTest : public ::testing::Test {
 protected:
  WARPLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 10, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 10, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    int dim = blob_bottom_label_->count() / blob_bottom_label_->num();
    Dtype* label_ptr = blob_bottom_label_->mutable_cpu_data();
    memset(label_ptr, 0, sizeof(Dtype) * blob_bottom_label_->count());
    int offset;
    int num_trials = std::max(std::min(dim, 3), 1);
    for (int i = 0; i < blob_bottom_label_->num(); ++i) {
      offset = i * dim;
      for (int j = 0; j < num_trials; ++j) {
        label_ptr[offset + rand() % dim] = 1;
      }
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
  }

  virtual ~WARPLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(WARPLossLayerTest, Dtypes);


TYPED_TEST(WARPLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  WARPLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(WARPLossLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  WARPLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

}
