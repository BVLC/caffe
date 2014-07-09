// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstring>
#include <cfloat>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class AccuracyLayerTest : public ::testing::Test {
 protected:
  AccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(2, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        top_k_(3) {
    // fill the probability values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

    // set the labels; first label is set to max-probability index
    Dtype m_val = -FLT_MAX;
    int m_id = 0;
    for (int i = 0; i < 10; i++)
      if (blob_bottom_data_->data_at(0, i, 0, 0) > m_val) {
        m_val = blob_bottom_data_->data_at(0, i, 0, 0);
        m_id = i;
      }
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    int offset = blob_bottom_label_->offset(0);
    label_data[offset] = m_id;

    // set the labels; second label is set to min-probability index
    m_val = FLT_MAX;
    m_id = 0;
    for (int i = 0; i < 10; i++)
      if (blob_bottom_data_->data_at(1, i, 0, 0) < m_val) {
        m_val = blob_bottom_data_->data_at(1, i, 0, 0);
        m_id = i;
      }
    offset = blob_bottom_label_->offset(1);
    label_data[offset] = m_id;

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~AccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int top_k_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(AccuracyLayerTest, Dtypes);

TYPED_TEST(AccuracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  AccuracyParameter* accuracy_param = layer_param.mutable_accuracy_param();
  accuracy_param->set_top_k(this->top_k_);
  Caffe::set_mode(Caffe::CPU);
  AccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));

  // Output accuracy should be ~0.5
  EXPECT_NEAR(this->blob_top_->data_at(0, 0, 0, 0), 0.5, 1e-4);
}

}  // namespace caffe
