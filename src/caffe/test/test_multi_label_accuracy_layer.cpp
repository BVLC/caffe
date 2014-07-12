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

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class MultiLabelAccuracyLayerTest : public ::testing::Test {
 protected:
  MultiLabelAccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // Fill the data vector
    FillerParameter data_filler_param;
    data_filler_param.set_std(1);
    GaussianFiller<Dtype> data_filler(data_filler_param);
    data_filler.Fill(blob_bottom_data_);

    // Fill the label vector
    FillerParameter targets_filler_param;
    targets_filler_param.set_min(-1);
    targets_filler_param.set_max(1);
    UniformFiller<Dtype> targets_filler(targets_filler_param);
    targets_filler.Fill(blob_bottom_label_);
    int count = blob_bottom_label_->count();
    caffe_cpu_sign(count, this->blob_bottom_label_->cpu_data(),
      this->blob_bottom_label_->mutable_cpu_data());

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~MultiLabelAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
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
    Dtype mean_acc = 0;
    int num_repetitions = 10;
    for (int i = 0; i < num_repetitions; ++i) {
      // Fill the data vector
      data_filler.Fill(this->blob_bottom_data_);
      // Fill the targets vector
      targets_filler.Fill(this->blob_bottom_label_);
      // Make negatives into -1 and positives into 1
      Dtype* targets = this->blob_bottom_label_->mutable_cpu_data();
      caffe_cpu_sign(count, targets, targets);
      // Add some 0s as in dropout
      caffe_rng_bernoulli(count, 1. - threshold_, mask);
      for (int j = 0; j < count; ++j) {
        targets[j] = targets[j] * mask[j];
      }
      MultiLabelAccuracyLayer<Dtype> layer(layer_param);
      layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
      layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
      const Dtype* top_acc = this->blob_top_->cpu_data();
      DLOG(INFO) << top_acc[0] << " " << top_acc[1] << " " << top_acc[2];
      mean_acc += top_acc[2];
    }
    return mean_acc/num_repetitions;
  }
  shared_ptr<SyncedMemory> rand_vec_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MultiLabelAccuracyLayerTest, Dtypes);

TYPED_TEST(MultiLabelAccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  MultiLabelAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 5);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithoutZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should use all the samples, since no label = 0
  // The mean_acc should be positive and greater than 1
  TypeParam mean_acc = this->TestForward(TypeParam(0));
  EXPECT_GT(mean_acc, 0);
  EXPECT_LT(mean_acc, 1);
}

TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithHalfZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should use half of the samples, since half of labels are 0
  // The mean_acc should be positive and greater than 1
  TypeParam mean_acc = this->TestForward(TypeParam(0.5));
  EXPECT_GT(mean_acc, 0);
  EXPECT_LT(mean_acc, 1);
}

TYPED_TEST(MultiLabelAccuracyLayerTest, TestWithAllZeros) {
  Caffe::set_mode(Caffe::CPU);
  // Should ignore all the samples, since all labels 0
  // The mean_acc should be 0 when we ignore all the labels
  TypeParam mean_acc = this->TestForward(TypeParam(1));
  EXPECT_EQ(mean_acc, 0);
}

}  // namespace caffe
