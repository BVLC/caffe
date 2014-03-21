// Copyright 2014 kloudkl@github
// Copyright 2014 Jeff Donahue
// Copyright 2014 Alejandro Dubrovsky
// Copyright 2014 Evan Shelhamer

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class RandomNumberGeneratorTest : public ::testing::Test {
 public:
  virtual ~RandomNumberGeneratorTest() {}

  Dtype sample_mean(const Dtype* const seqs, const size_t sample_size) {
      double sum = 0;
      for (int i = 0; i < sample_size; ++i) {
          sum += seqs[i];
      }
      return sum / sample_size;
  }

  Dtype sample_mean(const int* const seqs, const size_t sample_size) {
      Dtype sum = 0;
      for (int i = 0; i < sample_size; ++i) {
          sum += Dtype(seqs[i]);
      }
      return sum / sample_size;
  }

  Dtype mean_bound(const Dtype std, const size_t sample_size) {
      return  std/sqrt(static_cast<double>(sample_size));
  }
};


typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(RandomNumberGeneratorTest, Dtypes);


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian) {
  size_t sample_size = 10000;
  SyncedMemory data_a(sample_size * sizeof(TypeParam));
  Caffe::set_random_seed(1701);
  TypeParam mu = 0;
  TypeParam sigma = 1;
  caffe_vRngGaussian(sample_size,
      reinterpret_cast<TypeParam*>(data_a.mutable_cpu_data()), mu, sigma);
  TypeParam true_mean = mu;
  TypeParam true_std = sigma;
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam empirical_mean =
      this->sample_mean(reinterpret_cast<const TypeParam*>(data_a.cpu_data()),
          sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform) {
  size_t sample_size = 10000;
  SyncedMemory data_a(sample_size * sizeof(TypeParam));
  Caffe::set_random_seed(1701);
  TypeParam lower = 0;
  TypeParam upper = 1;
  caffe_vRngUniform(sample_size,
      reinterpret_cast<TypeParam*>(data_a.mutable_cpu_data()), lower, upper);
  TypeParam true_mean = (lower + upper) / 2;
  TypeParam true_std = (upper - lower) / sqrt(12);
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam empirical_mean =
      this->sample_mean(reinterpret_cast<const TypeParam*>(data_a.cpu_data()),
          sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli) {
  size_t sample_size = 10000;
  SyncedMemory data_a(sample_size * sizeof(int));
  Caffe::set_random_seed(1701);
  double p = 0.3;
  caffe_vRngBernoulli(sample_size,
      static_cast<int*>(data_a.mutable_cpu_data()), p);
  TypeParam true_mean = p;
  TypeParam true_std = sqrt(p * (1 - p));
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam empirical_mean =
      this->sample_mean((const int *)data_a.cpu_data(), sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
}


}  // namespace caffe
