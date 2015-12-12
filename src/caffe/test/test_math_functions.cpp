#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <cmath>  // for std::fabs

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"

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

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
TYPED_TEST(CPUMathFunctionsTest, TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int dist;
  GetDevice<TypeParam>(Caffe::CPU)->hamming_distance(n, x, y, &dist);
  EXPECT_EQ(this->ReferenceHammingDistance(n, x, y), dist);
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
TYPED_TEST(CPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam cpu_asum;
  GetDevice<TypeParam>(Caffe::CPU)->asum(n, x, &cpu_asum);
  EXPECT_LT((cpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(CPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  GetDevice<TypeParam>(Caffe::CPU)->sign(n, x,
      this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  GetDevice<TypeParam>(Caffe::CPU)->sgnbit(n, x,
      this->blob_bottom_->mutable_cpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
=======
>>>>>>> pod/device/blob.hpp
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> origin/BVLC/parallel
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> origin/BVLC/parallel
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
  const TypeParam* abs_val = this->blob_bottom_->cpu_diff();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(abs_val[i], x[i] > 0 ? x[i] : -x[i]);
  }
}

TYPED_TEST(CPUMathFunctionsTest, TestScale) {
  int n = this->blob_bottom_->count();
  TypeParam alpha = this->blob_bottom_->cpu_diff()[caffe_rng_rand() %
                                                   this->blob_bottom_->count()];
  GetDevice<TypeParam>(Caffe::CPU)->scale(n, alpha,
      this->blob_bottom_->cpu_data(), this->blob_bottom_->mutable_cpu_diff());
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  caffe_copy(n, bottom_data, top_data);
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
// TODO: Fix caffe_gpu_hamming_distance and re-enable this test.
TYPED_TEST(GPUMathFunctionsTest, DISABLED_TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->gpu_data();
  y = this->blob_top_->gpu_data();
  int computed_distance;
  GetDevice<TypeParam>(Caffe::GPU)->hamming_distance(n, x, y,
      &computed_distance);
  EXPECT_EQ(reference_distance, computed_distance);
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  GetDevice<TypeParam>(Caffe::GPU)->asum(n, this->blob_bottom_->gpu_data(),
                                         &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  GetDevice<TypeParam>(Caffe::GPU)->sign(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
  GetDevice<TypeParam>(Caffe::GPU)->sgnbit(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
<<<<<<< HEAD
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
<<<<<<< HEAD
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> pod/device/blob.hpp
=======
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->fabs(n, x,
      this->blob_bottom_->mutable_cpu_diff());
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> BVLC/master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> master
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> origin/BVLC/parallel
=======
  caffe_abs<TypeParam>(n, x, this->blob_bottom_->mutable_cpu_diff());
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
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
  GetDevice<TypeParam>(Caffe::GPU)->scale(n, alpha,
      this->blob_bottom_->gpu_data(), this->blob_bottom_->mutable_gpu_diff());
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  caffe_copy(n, bottom_data, top_data);
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::CPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
class GPUMathFunctionsTest : public MathFunctionsTest<GPUDevice<Dtype> > {
};

TYPED_TEST_CASE(GPUMathFunctionsTest, TestDtypes);

// TODO: Fix caffe_gpu_hamming_distance and re-enable this test.
TYPED_TEST(GPUMathFunctionsTest, DISABLED_TestHammingDistance) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  const TypeParam* y = this->blob_top_->cpu_data();
  int reference_distance = this->ReferenceHammingDistance(n, x, y);
  x = this->blob_bottom_->gpu_data();
  y = this->blob_top_->gpu_data();
  int computed_distance;
  GetDevice<TypeParam>(Caffe::GPU)->hamming_distance(n, x, y,
      &computed_distance);
  EXPECT_EQ(reference_distance, computed_distance);
}

TYPED_TEST(GPUMathFunctionsTest, TestAsum) {
  int n = this->blob_bottom_->count();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  TypeParam std_asum = 0;
  for (int i = 0; i < n; ++i) {
    std_asum += std::fabs(x[i]);
  }
  TypeParam gpu_asum;
  GetDevice<TypeParam>(Caffe::GPU)->asum(n, this->blob_bottom_->gpu_data(),
                                         &gpu_asum);
  EXPECT_LT((gpu_asum - std_asum) / std_asum, 1e-2);
}

TYPED_TEST(GPUMathFunctionsTest, TestSign) {
  int n = this->blob_bottom_->count();
  GetDevice<TypeParam>(Caffe::GPU)->sign(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signs = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signs[i], x[i] > 0 ? 1 : (x[i] < 0 ? -1 : 0));
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestSgnbit) {
  int n = this->blob_bottom_->count();
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> BVLC/device-abstraction
  GetDevice<TypeParam>(Caffe::GPU)->sgnbit(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
  const TypeParam* signbits = this->blob_bottom_->cpu_diff();
  const TypeParam* x = this->blob_bottom_->cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(signbits[i], x[i] < 0 ? 1 : 0);
  }
}

TYPED_TEST(GPUMathFunctionsTest, TestFabs) {
  int n = this->blob_bottom_->count();
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
=======
>>>>>>> pod/device/blob.hpp
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  caffe_gpu_abs<TypeParam>(n, this->blob_bottom_->gpu_data(),
                            this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::GPU)->fabs(n, this->blob_bottom_->gpu_data(),
      this->blob_bottom_->mutable_gpu_diff());
>>>>>>> BVLC/device-abstraction
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
  GetDevice<TypeParam>(Caffe::GPU)->scale(n, alpha,
      this->blob_bottom_->gpu_data(), this->blob_bottom_->mutable_gpu_diff());
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  caffe_copy(n, bottom_data, top_data);
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/device/blob.hpp
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod/caffe-merge
=======
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
  caffe_copy(n, bottom_data, top_data);
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
  GetDevice<TypeParam>(Caffe::GPU)->copy(n, bottom_data, top_data);
>>>>>>> BVLC/device-abstraction
  bottom_data = this->blob_bottom_->cpu_data();
  top_data = this->blob_top_->mutable_cpu_data();
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(bottom_data[i], top_data[i]);
  }
}

#endif


}  // namespace caffe
