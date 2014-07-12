// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
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

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class MultiLabelLossLayerTest : public ::testing::Test {
 protected:
  MultiLabelLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_targets_(new Blob<Dtype>(10, 5, 1, 1)) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    // Fill the targets vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_targets_);
    int count = blob_bottom_targets_->count();
    caffe_cpu_sign(count, this->blob_bottom_targets_->cpu_data(),
      this->blob_bottom_targets_->mutable_cpu_data());
    blob_bottom_vec_.push_back(blob_bottom_targets_);
  }
  virtual ~MultiLabelLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
  }

  Dtype SigmoidMultiLabelLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    for (int i = 0; i < count; ++i) {
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], -1);
      if (target[i] != 0) {
        loss -= (target[i] > 0) * log(prediction + (target[i] < 0));
        loss -= (target[i] < 0) * log(1 - prediction + (target[i] > 0));
      }
    }
    return loss / num;
  }

  void TestForward() {
    LayerParameter layer_param;
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    const int count = this->blob_bottom_data_->count();
    Dtype eps = 2e-2;
    for (int i = 0; i < 10; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      // Make negatives into -1 and positives into 1
      Dtype* targets = this->blob_bottom_targets_->mutable_cpu_data();
      caffe_cpu_sign(count, targets, targets);
      MultiLabelLossLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = this->SigmoidMultiLabelLossReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
    }
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MultiLabelLossLayerTest, Dtypes);

TYPED_TEST(MultiLabelLossLayerTest, TestSetup1Top) {
  LayerParameter layer_param;
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  vector<Blob<TypeParam>*> aux_top_vec;
  Blob<TypeParam>* blob_top_ = new Blob<TypeParam>();
  aux_top_vec.push_back(blob_top_);
  layer.SetUp(this->blob_bottom_vec_, &(aux_top_vec));
  EXPECT_EQ(blob_top_->num(), 1);
  EXPECT_EQ(blob_top_->channels(), 1);
  EXPECT_EQ(blob_top_->height(), 1);
  EXPECT_EQ(blob_top_->width(), 1);
}

TYPED_TEST(MultiLabelLossLayerTest, TestSetup2Tops) {
  LayerParameter layer_param;
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  vector<Blob<TypeParam>*> aux_top_vec;
  Blob<TypeParam>* blob_top_ = new Blob<TypeParam>();
  Blob<TypeParam>* blob_top2_ = new Blob<TypeParam>();
  aux_top_vec.push_back(blob_top_);
  aux_top_vec.push_back(blob_top2_);
  layer.SetUp(this->blob_bottom_vec_, &(aux_top_vec));
  EXPECT_EQ(blob_top_->num(), 1);
  EXPECT_EQ(blob_top_->channels(), 1);
  EXPECT_EQ(blob_top_->height(), 1);
  EXPECT_EQ(blob_top_->width(), 1);
  EXPECT_EQ(blob_top2_->num(), this->blob_bottom_targets_->num());
  EXPECT_EQ(blob_top2_->channels(), this->blob_bottom_targets_->channels());
  EXPECT_EQ(blob_top2_->height(), this->blob_bottom_targets_->height());
  EXPECT_EQ(blob_top2_->width(), this->blob_bottom_targets_->width());
}

TYPED_TEST(MultiLabelLossLayerTest, TestSigmoidCrossEntropyLossCPU) {
  Caffe::set_mode(Caffe::CPU);
  this->TestForward();
}

TYPED_TEST(MultiLabelLossLayerTest, TestSigmoidCrossEntropyLossGPU) {
  Caffe::set_mode(Caffe::GPU);
  this->TestForward();
}

TYPED_TEST(MultiLabelLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

TYPED_TEST(MultiLabelLossLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}


}  // namespace caffe
