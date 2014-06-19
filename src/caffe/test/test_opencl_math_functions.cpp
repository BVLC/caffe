// Copyright 2014 BVLC and contributors.

#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <climits>
#include <cmath>  // for std::fabs
#include <cstdlib>  // for rand_r

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/opencl_math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template<typename Dtype>
class OpenCLMathFunctionsTest : public ::testing::Test {
 protected:
  OpenCLMathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_bottom2_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom2_);
    filler.Fill(this->blob_top_);
  }

  virtual ~OpenCLMathFunctionsTest() {
    delete blob_bottom_;
    delete blob_bottom2_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_top_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(OpenCLMathFunctionsTest, Dtypes);

TYPED_TEST(OpenCLMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

// TODO: Fix caffe_opencl_hamming_distance and re-enable this test.
TYPED_TEST(OpenCLMathFunctionsTest, DISABLED_TestHammingDistanceOpenCL) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->opencl_data();
  y = this->blob_top_->opencl_data();
  int computed_distance = caffe_opencl_hamming_distance<TypeParam>(n, x, y);
  EXPECT_EQ(reference_distance, computed_distance);
}

TYPED_TEST(OpenCLMathFunctionsTest, TestAsumOpenCL) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam opencl_asum;
  caffe_opencl_asum<TypeParam>(n, this->blob_bottom_->opencl_data(), &opencl_asum);
  EXPECT_LT((opencl_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSignOpenCL) {
  int n = this->blob_bottom_->count();
  caffe_opencl_sign<TypeParam>(n, this->blob_bottom_->opencl_data(),
                            this->blob_bottom_->mutable_opencl_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSgnbitOpenCL) {
  int n = this->blob_bottom_->count();
  caffe_opencl_sgnbit<TypeParam>(n, this->blob_bottom_->opencl_data(),
                            this->blob_bottom_->mutable_opencl_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestFabsOpenCL) {
  int n = this->blob_bottom_->count();
  caffe_opencl_fabs<TypeParam>(n, this->blob_bottom_->opencl_data(),
                            this->blob_bottom_->mutable_opencl_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestScaleOpenCL) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_opencl_scale<TypeParam>(n, alpha, this->blob_bottom_->opencl_data(),
                             this->blob_bottom_->mutable_opencl_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestCopyFromCPU) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  caffe_opencl_copy_from_cpu(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestCopyOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSqrOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_sqr(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestExpOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_exp(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSignOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_sign(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSgnbitOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_sgnbit(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestFabsOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_fabs(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestAddOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  const TypeParam* bottom2_data = this->blob_bottom2_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_add(n, bottom_data, bottom2_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  bottom2_data = this->blob_bottom2_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i] + bottom2_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestSubOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  const TypeParam* bottom2_data = this->blob_bottom2_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_sub(n, bottom_data, bottom2_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  bottom2_data = this->blob_bottom2_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i] - bottom2_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestMulOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  const TypeParam* bottom2_data = this->blob_bottom2_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_mul(n, bottom_data, bottom2_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  bottom2_data = this->blob_bottom2_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i] * bottom2_data[i], top_data[i]);
  }
}

TYPED_TEST(OpenCLMathFunctionsTest, TestDivOpenCL) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->opencl_data();
  const TypeParam* bottom2_data = this->blob_bottom2_->opencl_data();
  TypeParam* top_data = this->blob_top_->mutable_opencl_data();
  caffe_opencl_div(n, bottom_data, bottom2_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  bottom2_data = this->blob_bottom2_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i] / std::min(bottom2_data[i], 1e-5), top_data[i]);
  }
}

}  // namespace caffe
