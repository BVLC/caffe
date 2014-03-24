// Copyright 2014 BVLC and contributors.

#include <cstring>

#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

typedef ::testing::Types<float, double> Dtypes;

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~ConstantFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, Dtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.value());
  }
}


template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
  UniformFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~UniformFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, Dtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.min());
    EXPECT_LE(data[i], this->filler_param_.max());
  }
}

template <typename Dtype>
class PositiveUnitballFillerTest : public ::testing::Test {
 protected:
  PositiveUnitballFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~PositiveUnitballFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballFillerTest, Dtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int num = this->blob_->num();
  const int count = this->blob_->count();
  const int dim = count / num;
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LE(data[i], 1);
  }
  for (int i = 0; i < num; ++i) {
    TypeParam sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += data[i * dim + j];
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~GaussianFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, Dtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  TypeParam mean = 0.;
  TypeParam var = 0.;
  for (int i = 0; i < count; ++i) {
    mean += data[i];
    var += (data[i] - this->filler_param_.mean()) *
        (data[i] - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  // Very loose test.
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  TypeParam target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);
}

}  // namespace caffe
