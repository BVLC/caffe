// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ArgMaxLayerTest : public ::testing::Test {
 protected:
  ArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 20, 1, 1)),
        blob_top_(new Blob<Dtype>()),
        top_k_(5) {
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ArgMaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;
};

TYPED_TEST_CASE(ArgMaxLayerTest, TestDtypes);

TYPED_TEST(ArgMaxLayerTest, TestSetup) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
}

TYPED_TEST(ArgMaxLayerTest, TestSetupMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 2);
}

TYPED_TEST(ArgMaxLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i];
    max_val = bottom_data[i * dim + max_ind];
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxVal) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2];
    max_val = top_data[i * 2 + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
    }
  }
}

template<typename Dtype>
bool int_Dtype_pair_greater(std::pair<int, Dtype> a,
                            std::pair<int, Dtype> b) {
  return a.second > b.second || (a.second == b.second && a.first > b.first);
}

TYPED_TEST(ArgMaxLayerTest, TestCPUTopK) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * this->top_k_];
    max_val = bottom_data[i * dim + max_ind];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    std::vector<std::pair<int, TypeParam> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
      bottom_data_vector.push_back(std::make_pair(j, bottom_data[i * dim + j]));
    }
    std::sort(bottom_data_vector.begin(), bottom_data_vector.end(),
              int_Dtype_pair_greater<TypeParam>);
    for (int j = 0; j < this->top_k_; ++j) {
      EXPECT_EQ(bottom_data_vector[j].first, top_data[i * this->top_k_ + j]);
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxValTopK) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  argmax_param->set_top_k(this->top_k_);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Now, check values
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int max_ind;
  TypeParam max_val;
  int num = this->blob_bottom_->num();
  int dim = this->blob_bottom_->count() / num;
  for (int i = 0; i < num; ++i) {
    EXPECT_GE(top_data[i], 0);
    EXPECT_LE(top_data[i], dim);
    max_ind = top_data[i * 2 * this->top_k_];
    max_val = top_data[i * 2 * this->top_k_ + 1];
    EXPECT_EQ(bottom_data[i * dim + max_ind], max_val);
    std::vector<std::pair<int, TypeParam> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      EXPECT_LE(bottom_data[i * dim + j], max_val);
      bottom_data_vector.push_back(std::make_pair(j, bottom_data[i * dim + j]));
    }
    std::sort(bottom_data_vector.begin(), bottom_data_vector.end(),
              int_Dtype_pair_greater<TypeParam>);
    for (int j = 0; j < this->top_k_; ++j) {
      EXPECT_EQ(bottom_data_vector[j].first,
                top_data[i * 2 * this->top_k_ + 2 * j]);
      EXPECT_EQ(bottom_data_vector[j].second,
                top_data[i * 2 * this->top_k_ + 2 * j + 1]);
    }
  }
}


}  // namespace caffe
