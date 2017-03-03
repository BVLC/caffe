/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/dummy_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class DummyDataLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  DummyDataLayerTest()
      : blob_top_a_(new Blob<Dtype>()),
        blob_top_b_(new Blob<Dtype>()),
        blob_top_c_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_bottom_vec_.clear();
    blob_top_vec_.clear();
    blob_top_vec_.push_back(blob_top_a_);
    blob_top_vec_.push_back(blob_top_b_);
    blob_top_vec_.push_back(blob_top_c_);
  }

  virtual ~DummyDataLayerTest() {
    delete blob_top_a_;
    delete blob_top_b_;
    delete blob_top_c_;
  }

  Blob<Dtype>* const blob_top_a_;
  Blob<Dtype>* const blob_top_b_;
  Blob<Dtype>* const blob_top_c_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DummyDataLayerTest, TestDtypes);

TYPED_TEST(DummyDataLayerTest, TestOneTopConstant) {
  LayerParameter param;
  DummyDataParameter* dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_num(5);
  dummy_data_param->add_channels(3);
  dummy_data_param->add_height(2);
  dummy_data_param->add_width(4);
  this->blob_top_vec_.resize(1);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->num(), 5);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 2);
  EXPECT_EQ(this->blob_top_a_->width(), 4);
  EXPECT_EQ(this->blob_top_b_->count(), 0);
  EXPECT_EQ(this->blob_top_c_->count(), 0);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataLayerTest, TestTwoTopConstant) {
  LayerParameter param;
  DummyDataParameter* dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_num(5);
  dummy_data_param->add_channels(3);
  dummy_data_param->add_height(2);
  dummy_data_param->add_width(4);
  dummy_data_param->add_num(5);
  // Don't explicitly set number of channels or height for 2nd top blob; should
  // default to first channels and height (as we check later).
  dummy_data_param->add_height(1);
  FillerParameter* data_filler_param = dummy_data_param->add_data_filler();
  data_filler_param->set_value(7);
  this->blob_top_vec_.resize(2);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->num(), 5);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 2);
  EXPECT_EQ(this->blob_top_a_->width(), 4);
  EXPECT_EQ(this->blob_top_b_->num(), 5);
  EXPECT_EQ(this->blob_top_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_b_->height(), 1);
  EXPECT_EQ(this->blob_top_b_->width(), 4);
  EXPECT_EQ(this->blob_top_c_->count(), 0);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataLayerTest, TestThreeTopConstantGaussianConstant) {
  LayerParameter param;
  DummyDataParameter* dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_num(5);
  dummy_data_param->add_channels(3);
  dummy_data_param->add_height(2);
  dummy_data_param->add_width(4);
  FillerParameter* data_filler_param_a = dummy_data_param->add_data_filler();
  data_filler_param_a->set_value(7);
  FillerParameter* data_filler_param_b = dummy_data_param->add_data_filler();
  data_filler_param_b->set_type("gaussian");
  TypeParam gaussian_mean = 3.0;
  TypeParam gaussian_std = 0.01;
  data_filler_param_b->set_mean(gaussian_mean);
  data_filler_param_b->set_std(gaussian_std);
  FillerParameter* data_filler_param_c = dummy_data_param->add_data_filler();
  data_filler_param_c->set_value(9);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_a_->num(), 5);
  EXPECT_EQ(this->blob_top_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_a_->height(), 2);
  EXPECT_EQ(this->blob_top_a_->width(), 4);
  EXPECT_EQ(this->blob_top_b_->num(), 5);
  EXPECT_EQ(this->blob_top_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_b_->height(), 2);
  EXPECT_EQ(this->blob_top_b_->width(), 4);
  EXPECT_EQ(this->blob_top_c_->num(), 5);
  EXPECT_EQ(this->blob_top_c_->channels(), 3);
  EXPECT_EQ(this->blob_top_c_->height(), 2);
  EXPECT_EQ(this->blob_top_c_->width(), 4);
  for (int i = 0; i < this->blob_top_a_->count(); ++i) {
    EXPECT_EQ(7, this->blob_top_a_->cpu_data()[i]);
  }
  // Blob b uses a Gaussian filler, so SetUp should not have initialized it.
  // Blob b's data should therefore be the default Blob data value: 0.
  for (int i = 0; i < this->blob_top_b_->count(); ++i) {
    EXPECT_EQ(0, this->blob_top_b_->cpu_data()[i]);
  }
  for (int i = 0; i < this->blob_top_c_->count(); ++i) {
    EXPECT_EQ(9, this->blob_top_c_->cpu_data()[i]);
  }

  // Do a Forward pass to fill in Blob b with Gaussian data.
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_a_->count(); ++i) {
    EXPECT_EQ(7, this->blob_top_a_->cpu_data()[i]);
  }
  // Check that the Gaussian's data has been filled in with values within
  // 10 standard deviations of the mean. Record the first and last sample.
  // to check that they're different after the next Forward pass.
  for (int i = 0; i < this->blob_top_b_->count(); ++i) {
    EXPECT_NEAR(gaussian_mean, this->blob_top_b_->cpu_data()[i],
                gaussian_std * 10);
  }
  const TypeParam first_gaussian_sample = this->blob_top_b_->cpu_data()[0];
  const TypeParam last_gaussian_sample =
      this->blob_top_b_->cpu_data()[this->blob_top_b_->count() - 1];
  for (int i = 0; i < this->blob_top_c_->count(); ++i) {
    EXPECT_EQ(9, this->blob_top_c_->cpu_data()[i]);
  }

  // Do another Forward pass to fill in Blob b with Gaussian data again,
  // checking that we get different values.
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_a_->count(); ++i) {
    EXPECT_EQ(7, this->blob_top_a_->cpu_data()[i]);
  }
  for (int i = 0; i < this->blob_top_b_->count(); ++i) {
    EXPECT_NEAR(gaussian_mean, this->blob_top_b_->cpu_data()[i],
                gaussian_std * 10);
  }
  EXPECT_NE(first_gaussian_sample, this->blob_top_b_->cpu_data()[0]);
  EXPECT_NE(last_gaussian_sample,
      this->blob_top_b_->cpu_data()[this->blob_top_b_->count() - 1]);
  for (int i = 0; i < this->blob_top_c_->count(); ++i) {
    EXPECT_EQ(9, this->blob_top_c_->cpu_data()[i]);
  }
}

}  // namespace caffe
