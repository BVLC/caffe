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

#include "gtest/gtest.h"

#include "caffe/filler.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

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

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

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

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

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

TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

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

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

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

template <typename Dtype>
class XavierFillerTest : public ::testing::Test {
 protected:
  XavierFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new XavierFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~XavierFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<XavierFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(XavierFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(XavierFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename Dtype>
class GaborFillerTest : public ::testing::Test {
 protected:
  GaborFillerTest()
      : blob_(new Blob<Dtype>(96, 3, 11, 11)),
        filler_param_() {
  }
  virtual void test_no_crash() {
    this->filler_.reset(new GaborFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    // const int count = this->blob_->count();
    // const Dtype* data = this->blob_->cpu_data();
  }

  virtual ~GaborFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaborFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaborFillerTest, TestDtypes);

TYPED_TEST(GaborFillerTest, TestNoCrash) {
  this->test_no_crash();
}

template <typename Dtype>
class MSRAFillerTest : public ::testing::Test {
 protected:
  MSRAFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new MSRAFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~MSRAFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<MSRAFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(MSRAFillerTest, TestDtypes);

TYPED_TEST(MSRAFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(MSRAFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(MSRAFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

}  // namespace caffe
