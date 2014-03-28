// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <vector>

#include "cuda_runtime.h"
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
class LRNMapLayerTest : public ::testing::Test {
 protected:
  LRNMapLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 2, 7, 7);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    epsilon_ = 1e-5;
  }
  virtual ~LRNMapLayerTest() { delete blob_bottom_; delete blob_top_; }
  void ReferenceLRNMapForward(const Blob<Dtype>& blob_bottom,
      const LayerParameter& layer_param, Blob<Dtype>* blob_top);
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  Dtype epsilon_;
};

template <typename Dtype>
void LRNMapLayerTest<Dtype>::ReferenceLRNMapForward(
    const Blob<Dtype>& blob_bottom, const LayerParameter& layer_param,
    Blob<Dtype>* blob_top) {
  blob_top->Reshape(blob_bottom.num(), blob_bottom.channels(),
      blob_bottom.height(), blob_bottom.width());
  const Dtype* bottom_data = blob_bottom.cpu_data();
  Dtype* top_data = blob_top->mutable_cpu_data();
  const Dtype alpha = layer_param.lrn_map_param().alpha();
  const Dtype beta = layer_param.lrn_map_param().beta();
  const int size = layer_param.lrn_map_param().local_size();
  for (int n = 0; n < blob_bottom.num(); ++n) {
    for (int c = 0; c < blob_bottom.channels(); ++c) {
      for (int h = 0; h < blob_bottom.height(); ++h) {
        int h_start = h - (size - 1) / 2;
        int h_end = min(h_start + size, blob_bottom.height());
        h_start = max(h_start, 0);
        for (int w = 0; w < blob_bottom.width(); ++w) {
          Dtype scale = 1.;
          int w_start = w - (size - 1) / 2;
          int w_end = min(w_start + size, blob_bottom.width());
          w_start = max(w_start, 0);
          for (int nh = h_start; nh < h_end; ++nh) {
            for (int nw = w_start; nw < w_end; ++nw) {
              Dtype value = blob_bottom.data_at(n, c, nh, nw);
              scale += value * value * alpha / (size * size);
            }
          }
          *(top_data + blob_top->offset(n, c, h, w)) =
            blob_bottom.data_at(n, c, h, w) / pow(scale, beta);
        }
      }
    }
  }
}

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(LRNMapLayerTest, Dtypes);

TYPED_TEST(LRNMapLayerTest, TestSetup) {
  LayerParameter layer_param;
  LRNMapLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), 7);
  EXPECT_EQ(this->blob_top_->width(), 7);
}

TYPED_TEST(LRNMapLayerTest, TestCPUForward) {
  LayerParameter layer_param;
  LRNMapLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Blob<TypeParam> top_reference;
  this->ReferenceLRNMapForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNMapLayerTest, TestGPUForward) {
  LayerParameter layer_param;
  LRNMapLayer<TypeParam> layer(layer_param);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  Blob<TypeParam> top_reference;
  this->ReferenceLRNMapForward(*(this->blob_bottom_), layer_param,
      &top_reference);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_top_->cpu_data()[i], top_reference.cpu_data()[i],
                this->epsilon_);
  }
}

TYPED_TEST(LRNMapLayerTest, TestCPUGradient) {
  LayerParameter layer_param;
  LRNMapLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  Caffe::set_mode(Caffe::CPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

TYPED_TEST(LRNMapLayerTest, TestGPUGradient) {
  LayerParameter layer_param;
  LRNMapLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  Caffe::set_mode(Caffe::GPU);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = 1.;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

}  // namespace caffe
