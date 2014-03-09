// Copyright 2013 Yangqing Jia

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
class LogisticRegressionLayerTest : public ::testing::Test {
 protected:
  LogisticRegressionLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1000, 1, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(1000, 1, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_min(0.02);
    filler_param.set_max(0.92);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
  }
  virtual ~LogisticRegressionLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(LogisticRegressionLayerTest, Dtypes);


TYPED_TEST(LogisticRegressionLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  LogisticRegressionLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-3, 1e-3, 1701);
  checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, -1, -1);
}

}
