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
class MultiLabelAccuracyLayerTest : public ::testing::Test {
 protected:
  MultiLabelAccuracyLayerTest()
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
  virtual ~MultiLabelAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_targets_;
  }

  Dtype SigmoidCrossEntropyLossReference(const int count, const int num,
                                         const Dtype* input,
                                         const Dtype* target) {
    Dtype loss = 0;
    int count_pos = 0;
    int count_neg = 0;
    int count_zeros = 0;
    for (int i = 0; i < count; ++i) {
      // It assumes -1 is negative, 1 is positive and 0 is ignore it.
      const Dtype prediction = 1 / (1 + exp(-input[i]));
      EXPECT_LE(prediction, 1);
      EXPECT_GE(prediction, 0);
      EXPECT_LE(target[i], 1);
      EXPECT_GE(target[i], -1);
      if (target[i] > 0) count_pos++;
      if (target[i] < 0) count_neg++;
      if (target[i] == 0) count_zeros++;
      loss -= (target[i] > 0) * log(prediction + (target[i] <= Dtype(0)));
      loss -= (target[i] < 0) * log(1 - prediction + (target[i] >= Dtype(0)));
    }
    // LOG(INFO) << "positives " << count_pos << " negatives " << count_neg <<
    //  " zeros " << count_zeros;
    return loss / num;
  }

  Dtype TestForward(Dtype threshold_ = Dtype(0)) {
    const int count = this->blob_bottom_data_->count();
    this->rand_vec_.reset(new SyncedMemory(count * sizeof(int)));
    int* mask = reinterpret_cast<int*>(this->rand_vec_->mutable_cpu_data());
    LayerParameter layer_param;
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1.0);
    targets_filler_param.set_max(1.0);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    Dtype eps = 2e-2;
    Dtype mean_loss = 0;
    int num_inf = 0;
    int num_repetitions = 100;
    for (int i = 0; i < num_repetitions; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_targets_);
      // Make negatives into -1 and positives into 1
      Dtype* targets = this->blob_bottom_targets_->mutable_cpu_data();
      caffe_cpu_sign(count, targets, targets);
      // Add some 0s as in dropout
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      for (int j = 0; j < count; ++j) {
        targets[j] = targets[j] * mask[j];
      }
      MultiLabelAccuracyLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
      Dtype layer_loss =
          layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
      const int num = this->blob_bottom_data_->num();
      const Dtype* blob_bottom_data = this->blob_bottom_data_->cpu_data();
      const Dtype* blob_bottom_targets =
          this->blob_bottom_targets_->cpu_data();
      Dtype reference_loss = this->SigmoidCrossEntropyLossReference(
          count, num, blob_bottom_data, blob_bottom_targets);
      EXPECT_NEAR(reference_loss, layer_loss, eps) << "debug: trial #" << i;
      mean_loss += layer_loss;
    }
    return mean_loss/num_repetitions;
  }
  shared_ptr<SyncedMemory> rand_vec_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_targets_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MultiLabelAccuracyLayerTest, Dtypes);


TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithoutZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should use all the samples, since no label = 0
  // The loss should be positive and greater than 1
  TypeParam loss = this->TestForward(TypeParam(0));
  CHECK_GE(loss, 1) << "loss should positive and greater than 1";
}

TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithHalfZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should use half of the samples, since half of labels are 0
  TypeParam loss = this->TestForward(TypeParam(0.5));
  CHECK_GE(loss, 0) << "loss should positive";
}

TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithAllZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should ignore all the samples, since all labels 0
  // The loss should be 0 when we ignore all the labels
  TypeParam eps = 2e-2;
  TypeParam loss = this->TestForward(TypeParam(1));
  CHECK_GE(loss, 0) << "loss should positive";
  EXPECT_NEAR(loss, eps) << "loss should be close to 0";
}

TYPED_TEST(MultiLabelAccuracyLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SigmoidCrossEntropyLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_), 0, -1, -1);
}

}  // namespace caffe
