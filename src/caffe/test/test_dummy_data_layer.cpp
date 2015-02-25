#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::string;
using std::stringstream;

namespace caffe {

template <typename Dtype>
class DummyDataLayerTest : public ::testing::Test {
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
  Caffe::set_mode(Caffe::CPU);
  LayerParameter param;
  DummyDataParameter* dummy_data_param = param.mutable_dummy_data_param();
  dummy_data_param->add_num(5);
  dummy_data_param->add_channels(3);
  dummy_data_param->add_height(2);
  dummy_data_param->add_width(4);
  this->blob_top_vec_.resize(1);
  DummyDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
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
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(0, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataLayerTest, TestTwoTopConstant) {
  Caffe::set_mode(Caffe::CPU);
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
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
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
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
  for (int i = 0; i < this->blob_top_vec_.size(); ++i) {
    for (int j = 0; j < this->blob_top_vec_[i]->count(); ++j) {
      EXPECT_EQ(7, this->blob_top_vec_[i]->cpu_data()[j]);
    }
  }
}

TYPED_TEST(DummyDataLayerTest, TestThreeTopConstantGaussianConstant) {
  Caffe::set_mode(Caffe::CPU);
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
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
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
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
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
  layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
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
