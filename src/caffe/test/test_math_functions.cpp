// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib> // for rand()
#include <cstring>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "test_math_functions_golden.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class MathFunctionsTest : public ::testing::Test {
 protected:
  MathFunctionsTest()
      : loops_(10)
        ,M_(12)
        ,N_(12)
        ,K_(15)
        ,a_(new Blob<Dtype>(2, 3, 6, 5))
        ,b_(new Blob<Dtype>(2, 3, 6, 5))
        ,y_(new Blob<Dtype>(2, 3, 6, 5))
        ,golden_y_(new Blob<Dtype>(2, 3, 6, 5))
        ,a_cpu_data_(a_->cpu_data())
        ,b_cpu_data_(b_->cpu_data())
        ,y_cpu_data_(y_->mutable_cpu_data())
        ,golden_y_cpu_data_(golden_y_->mutable_cpu_data())
        ,near_delta_(1e-5)
 {}

  virtual void SetUp() {
    num_ = a_->count();
    filler_param_.set_min(1e-5);
    filler_param_.set_max(10);
  }

  virtual ~MathFunctionsTest() {
    delete a_;
    delete b_;
    delete y_;
  }

  int loops_;
  int num_;
  int M_;
  int N_;
  int K_;
  Blob<Dtype>* a_;
  Blob<Dtype>* b_;
  Blob<Dtype>* y_;
  Blob<Dtype>* golden_y_;
  const Dtype* const a_cpu_data_;
  const Dtype* const b_cpu_data_;
  Dtype* y_cpu_data_;
  Dtype* golden_y_cpu_data_;
  const Dtype near_delta_;
  FillerParameter filler_param_;
  math_functions_cpu_golden<Dtype> golden_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MathFunctionsTest, Dtypes);

TYPED_TEST(MathFunctionsTest, TestAdd) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    filler.Fill(this->b_);
    caffe_add<TypeParam>(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
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
    caffe_sub<TypeParam>(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
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
    caffe_mul<TypeParam>(this->num_, this->a_->cpu_data(), this->b_->cpu_data(), this->y_cpu_data_);
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
    caffe_div<TypeParam>(this->num_, this->a_cpu_data_, this->b_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] /
                  this->b_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestPowx) {
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  TypeParam p;
  TypeParam ps[] = {-1.5, -0.5, 0, 0.5, 1.5};
  for (int l = 0; l < this->loops_; ++l) {
    for (int k = 0; k < 5; ++k) {
      p = ps[k];
      uniform_filler.Fill(this->a_);
      caffe_powx<TypeParam>(this->num_, this->a_cpu_data_, p, this->y_cpu_data_);
      for (int i = 0; i < this->num_; ++i) {
        EXPECT_NEAR(this->y_cpu_data_[i], std::pow(this->a_cpu_data_[i], p) ,
            this->near_delta_)
        << "debug: (i, y_cpu_data_, a_cpu_data_, p)="
        << i << "," << this->y_cpu_data_[i] << "," << this->a_cpu_data_[i]
        << "," << p;
      }
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestSqr) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    caffe_sqr<TypeParam>(this->num_, this->a_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->a_cpu_data_[i] * this->a_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestExp) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    caffe_exp<TypeParam>(this->num_, this->a_cpu_data_, this->y_cpu_data_);
    for (int i = 0; i < this->num_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], std::exp(this->a_cpu_data_[i]), this->near_delta_);
    }
  }
}


TYPED_TEST(MathFunctionsTest, TestCpuGemm) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    uniform_filler.Fill(this->b_);
    CBLAS_TRANSPOSE TransA;
    CBLAS_TRANSPOSE TransB;
    for (int ta = 0; ta < 2; ++ta) {
      TransA = ta ? CblasTrans : CblasNoTrans;
      for (int tb = 0; tb < 2; ++tb) {
        TransB = tb ? CblasTrans : CblasNoTrans;
        int alpha_idx = rand() % this->num_;
        int beta_idx = rand() % this->num_;
        caffe_cpu_gemm<TypeParam>(TransA,
            TransB, this->M_, this->N_, this->K_, this->a_cpu_data_[alpha_idx], this->a_cpu_data_,
            this->b_cpu_data_, this->b_cpu_data_[beta_idx], this->y_cpu_data_);
        this->golden_.gemm(TransA,
            TransB, this->M_, this->N_, this->K_, this->a_cpu_data_[alpha_idx], this->a_cpu_data_,
            this->b_cpu_data_, this->b_cpu_data_[beta_idx], this->golden_y_cpu_data_);
        for (int i = 0; i < this->num_; ++i) {
          EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
        }
      }
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestCpuGemv) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    CBLAS_TRANSPOSE TransA;
    for (int ta = 0; ta < 2; ++ta) {
      TransA = ta ? CblasTrans : CblasNoTrans;
      int alpha_idx = rand() % this->num_;
      int beta_idx = rand() % this->num_;
      filler.Fill(this->a_);
      uniform_filler.Fill(this->b_);
      filler.Fill(this->y_);
      for (int i = 0; i < this->num_; ++i) {
        this->golden_y_cpu_data_[i] = this->y_cpu_data_[i];
      }
      caffe_cpu_gemv<TypeParam>(TransA,
          this->M_, this->N_, this->a_cpu_data_[alpha_idx], this->a_cpu_data_,
          this->b_cpu_data_, this->b_cpu_data_[beta_idx], this->y_cpu_data_);
      this->golden_.gemv(TransA,
          this->M_, this->N_, this->a_cpu_data_[alpha_idx], this->a_cpu_data_,
          this->b_cpu_data_, this->b_cpu_data_[beta_idx], this->golden_y_cpu_data_);
      for (int i = 0; i < this->M_; ++i) {
        EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
      }
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestCpuAxpy) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    uniform_filler.Fill(this->a_);
    filler.Fill(this->b_);
    int alpha_idx = rand() % this->num_;
    caffe_axpy<TypeParam>(
        this->num_, this->a_cpu_data_[alpha_idx], this->b_cpu_data_, this->y_cpu_data_);
    this->golden_.axpy(this->num_, this->a_cpu_data_[alpha_idx], this->b_cpu_data_, this->golden_y_cpu_data_);
    for (int i = 0; i < this->M_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestCpuCopy) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    CBLAS_TRANSPOSE TransA;
    for (int ta = 0; ta < 2; ++ta) {
      TransA = ta ? CblasTrans : CblasNoTrans;
      caffe_copy<TypeParam>(
          this->num_, this->a_cpu_data_, this->y_cpu_data_);
      this->golden_.copy(this->num_, this->a_cpu_data_, this->golden_y_cpu_data_);
      for (int i = 0; i < this->M_; ++i) {
        EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
      }
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestCpuScal) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  UniformFiller<TypeParam> uniform_filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->y_);
    uniform_filler.Fill(this->a_);
    int alpha_idx = rand() % this->num_;
    for (int i = 0; i < this->M_; ++i) {
      this->golden_y_cpu_data_[i] = this->y_cpu_data_[i];
    }
    caffe_scal<TypeParam>(this->num_, this->a_cpu_data_[alpha_idx], this->y_cpu_data_);
    this->golden_.scal(this->num_, this->a_cpu_data_[alpha_idx], this->golden_y_cpu_data_);
    for (int i = 0; i < this->M_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
    }
  }
}

TYPED_TEST(MathFunctionsTest, TestCpuDot) {
  GaussianFiller<TypeParam> filler(this->filler_param_);
  for (int l = 0; l < this->loops_; ++l) {
    filler.Fill(this->a_);
    caffe_cpu_dot<TypeParam>(this->num_, this->a_cpu_data_, this->y_cpu_data_);
    this->golden_.dot(this->num_, this->a_cpu_data_, this->golden_y_cpu_data_);
    for (int i = 0; i < this->M_; ++i) {
      EXPECT_NEAR(this->y_cpu_data_[i], this->golden_y_cpu_data_[i], this->near_delta_);
    }
  }
}

}  // namespace caffe
