// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstring>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MathFunctionsTest : public ::testing::Test {
 protected:
  MathFunctionsTest()
      : loops_(10)
        ,a_(new Blob<Dtype>(2, 3, 6, 5))
        ,b_(new Blob<Dtype>(2, 3, 6, 5))
        ,y_(new Blob<Dtype>(2, 3, 6, 5))
        ,a_cpu_data_(a_->cpu_data())
        ,b_cpu_data_(b_->cpu_data())
        ,y_cpu_data_(y_->mutable_cpu_data())
        ,near_delta_(1e-5)
 {};

  virtual void SetUp() {
    num_ = a_->count();
    filler_param_.set_min(1e-5);
    filler_param_.set_max(10);
  };

  virtual ~MathFunctionsTest() {
    delete a_;
    delete b_;
    delete y_;
  }

  int loops_;
  int num_;
  Blob<Dtype>* a_;
  Blob<Dtype>* b_;
  Blob<Dtype>* y_;
  const Dtype* const a_cpu_data_;
  const Dtype* const b_cpu_data_;
  Dtype* y_cpu_data_;
  const Dtype near_delta_;
  FillerParameter filler_param_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MathFunctionsTest, Dtypes);

TYPED_TEST(MathFunctionsTest, TestAdd) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    filler.Fill(this->b_);
    caffe_add(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] + this->b_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestSub) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    filler.Fill(this->b_);
    caffe_sub(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] - this->b_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestMul) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    filler.Fill(this->b_);
    caffe_mul(this->num_, this->a_->cpu_data(), this->b_->cpu_data(), this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] * this->b_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestDiv) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    filler.Fill(this->b_);
    FillerParameter filler_param;
    filler_param.set_min(1e-5); // to avoid dividing by zero
    uniform_filler.Fill(this->b_);
    caffe_div(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] /
                  this->b_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestPowx) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  TypeParam p;
  for (int l = 0; l < this->loops_; ++l) {
    p = 0;
    filler.Fill(this->a_);
    caffe_powx(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
                  this->near_delta_)
       << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
       << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
       << "," << p;
    }

    p = 0.5;
    uniform_filler.Fill(this->a_);
    caffe_powx(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
                  this->near_delta_)
       << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
       << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
       << "," << p;
    }

    p = -0.5;
    uniform_filler.Fill(this->a_);
    caffe_powx(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
                  this->near_delta_)
       << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
       << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
       << "," << p;
    }

    p = 1.5;
    uniform_filler.Fill(this->a_);
    caffe_powx(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
                  this->near_delta_)
       << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
       << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
       << "," << p;
    }

    p = -1.5;
    uniform_filler.Fill(this->a_);
    caffe_powx(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
                  this->near_delta_)
       << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
       << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
       << "," << p;
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestSqr) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    caffe_sqr(this->num_, this->a_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] * this->a_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestExp) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    caffe_exp(this->num_, this->a_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::exp(this->a_cpu_data_[i]), this->near_delta_);
    }
  }
}

}  // namespace caffe
