#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <cmath>  // for std::fabs

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MathFunctionsTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MathFunctionsTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(11, 17, 19, 23);
    this->blob_top_->Reshape(11, 17, 19, 23);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MathFunctionsTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

template <typename Dtype>
class CPUMathFunctionsTest
  : public MathFunctionsTest<CPUDevice<Dtype> > {
};

TYPED_TEST_CASE(CPUMathFunctionsTest, TestDtypes);

TYPED_TEST(CPUMathFunctionsTest, TestNothing) {
  // The first test case of a test suite takes the longest time
  //   due to the set up overhead.
}

TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam cpu_asum = caffe_cpu_asum<TypeParam>(n, x);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sign<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_cpu_sgnbit<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_cpu_scale<TypeParam>(n, alpha, this->blob_bottom_->cpu_data(),
                             this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->cpu_data();
  TypeParam* top_data = this->blob_top_->mutable_cpu_data();
  caffe_copy(n, bottom_data, top_data);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  caffe_gpu_asum<TypeParam>(n, this->blob_bottom_->gpu_data(), &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sign<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  caffe_gpu_sgnbit<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  caffe_gpu_scale<TypeParam>(n, alpha, this->blob_bottom_->gpu_data(),
                             this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* scaled = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(scaled[i], x[i] * alpha);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestCopy) {
  const int n = this->blob_bottom_->count();
  const TypeParam* bottom_data = this->blob_bottom_->gpu_data();
  TypeParam* top_data = this->blob_top_->mutable_gpu_data();
  caffe_copy(n, bottom_data, top_data);
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#endif


}  // namespace caffe
