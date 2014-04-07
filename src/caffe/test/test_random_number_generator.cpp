// Copyright 2014 BVLC and contributors.

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


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesBernoulli) {
  size_t sample_size = 10000;
  SyncedMemory gaussian_data(sample_size * sizeof(TypeParam));
  SyncedMemory bernoulli_data(sample_size * sizeof(int));
  Caffe::set_random_seed(1701);
  // Sample from 0 mean Gaussian
  TypeParam mu = 0;
  TypeParam sigma = 1;
  caffe_vRngGaussian(sample_size, reinterpret_cast<TypeParam*>(
      gaussian_data.mutable_cpu_data()), mu, sigma);
  TypeParam true_mean = mu;
  TypeParam true_std = sigma;
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam empirical_mean = this->sample_mean(
      reinterpret_cast<const TypeParam*>(gaussian_data.cpu_data()),
      sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  int num_pos = 0;
  int num_neg = 0;
  int num_zeros = 0;
  TypeParam* samples =
      static_cast<TypeParam*>(gaussian_data.mutable_cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (samples[i] == TypeParam(0)) {
      ++num_zeros;
    } else if (samples[i] > TypeParam(0)) {
      ++num_pos;
    } else if (samples[i] < TypeParam(0)) {
      ++num_neg;
    }
  }
  // Check that we have no zeros (possible to generate 0s, but highly
  // improbable), and roughly half positives and half negatives (with bound
  // computed from a Bernoulli with p = 0.5).
  EXPECT_EQ(0, num_zeros);
  double p = 0.5;
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  bound = this->mean_bound(true_std, sample_size);
  TypeParam expected_num_each_sign = sample_size * p;
  LOG(INFO) << "Gaussian: Expected " << expected_num_each_sign << " positives"
            << "; got " << num_pos;
  LOG(INFO) << "Gaussian: Expected " << expected_num_each_sign << " negatives"
            << "; got " << num_neg;
  EXPECT_NEAR(expected_num_each_sign, num_pos, sample_size * bound);
  EXPECT_NEAR(expected_num_each_sign, num_neg, sample_size * bound);
  // Sample from Bernoulli with p = 0.3
  p = 0.3;
  caffe_vRngBernoulli(sample_size,
      reinterpret_cast<int*>(bernoulli_data.mutable_cpu_data()), p);
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  bound = this->mean_bound(true_std, sample_size);
  empirical_mean =
      this->sample_mean((const int *)bernoulli_data.cpu_data(), sample_size);
  LOG(INFO) << "Bernoulli: Expected mean = " << true_mean
            << "; sample mean = " << empirical_mean;
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  int bernoulli_num_zeros = 0;
  int num_ones = 0;
  int num_other = 0;
  const int* bernoulli_samples =
      reinterpret_cast<const int*>(bernoulli_data.cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (bernoulli_samples[i] == 0) {
      ++bernoulli_num_zeros;
    } else if (bernoulli_samples[i] == 1) {
      ++num_ones;
    } else {
      ++num_other;
    }
  }
  LOG(INFO) << "Bernoulli: zeros: "  << bernoulli_num_zeros
            << "; ones: " << num_ones << "; other: " << num_other;
  EXPECT_EQ(0, num_other);
  EXPECT_EQ(sample_size * empirical_mean, num_ones);
  EXPECT_EQ(sample_size * (1.0 - empirical_mean), bernoulli_num_zeros);
  // Multiply Gaussian by Bernoulli
  for (int i = 0; i < sample_size; ++i) {
    samples[i] *= bernoulli_samples[i];
  }
  num_pos = 0;
  num_neg = 0;
  num_zeros = 0;
  for (int i = 0; i < sample_size; ++i) {
    if (samples[i] == TypeParam(0)) {
      ++num_zeros;
    } else if (samples[i] > TypeParam(0)) {
      ++num_pos;
    } else if (samples[i] < TypeParam(0)) {
      ++num_neg;
    }
  }
  // Check that we have as many zeros as Bernoulli, and roughly half positives
  // and half negatives (with bound computed from a Bernoulli with p = 0.5).
  EXPECT_EQ(bernoulli_num_zeros, num_zeros);
  p = 0.5;
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  int sub_sample_size = sample_size - bernoulli_num_zeros;
  bound = this->mean_bound(true_std, sub_sample_size);
  expected_num_each_sign = sub_sample_size * p;
  LOG(INFO) << "Gaussian*Bernoulli: Expected " << expected_num_each_sign
            << " positives; got " << num_pos;
  LOG(INFO) << "Gaussian*Bernoulli: Expected " << expected_num_each_sign
            << " negatives; got " << num_neg;
  EXPECT_NEAR(expected_num_each_sign, num_pos, sample_size * bound);
  EXPECT_NEAR(expected_num_each_sign, num_neg, sample_size * bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesBernoulli) {
  size_t sample_size = 10000;
  SyncedMemory uniform_data(sample_size * sizeof(TypeParam));
  SyncedMemory bernoulli_data(sample_size * sizeof(int));
  Caffe::set_random_seed(1701);
  // Sample from Uniform on [-1, 1]
  TypeParam a = -1;
  TypeParam b = 1;
  caffe_vRngUniform(sample_size, reinterpret_cast<TypeParam*>(
      uniform_data.mutable_cpu_data()), a, b);
  TypeParam true_mean = (a + b) / 2;
  TypeParam true_std = (b - a) / sqrt(12);
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam empirical_mean = this->sample_mean(
      reinterpret_cast<const TypeParam*>(uniform_data.cpu_data()),
      sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  int num_pos = 0;
  int num_neg = 0;
  int num_zeros = 0;
  TypeParam* samples =
      static_cast<TypeParam*>(uniform_data.mutable_cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (samples[i] == TypeParam(0)) {
      ++num_zeros;
    } else if (samples[i] > TypeParam(0)) {
      ++num_pos;
    } else if (samples[i] < TypeParam(0)) {
      ++num_neg;
    }
  }
  // Check that we have no zeros (possible to generate 0s, but highly
  // improbable), and roughly half positives and half negatives (with bound
  // computed from a Bernoulli with p = 0.5).
  EXPECT_EQ(0, num_zeros);
  double p = 0.5;
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  bound = this->mean_bound(true_std, sample_size);
  TypeParam expected_num_each_sign = sample_size * p;
  LOG(INFO) << "Uniform: Expected " << expected_num_each_sign << " positives"
            << "; got " << num_pos;
  LOG(INFO) << "Uniform: Expected " << expected_num_each_sign << " negatives"
            << "; got " << num_neg;
  EXPECT_NEAR(expected_num_each_sign, num_pos, sample_size * bound);
  EXPECT_NEAR(expected_num_each_sign, num_neg, sample_size * bound);
  // Sample from Bernoulli with p = 0.3
  p = 0.3;
  caffe_vRngBernoulli(sample_size,
      reinterpret_cast<int*>(bernoulli_data.mutable_cpu_data()), p);
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  bound = this->mean_bound(true_std, sample_size);
  empirical_mean =
      this->sample_mean((const int *)bernoulli_data.cpu_data(), sample_size);
  LOG(INFO) << "Bernoulli: Expected mean = " << true_mean
            << "; sample mean = " << empirical_mean;
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  int bernoulli_num_zeros = 0;
  int num_ones = 0;
  int num_other = 0;
  const int* bernoulli_samples =
      reinterpret_cast<const int*>(bernoulli_data.cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (bernoulli_samples[i] == 0) {
      ++bernoulli_num_zeros;
    } else if (bernoulli_samples[i] == 1) {
      ++num_ones;
    } else {
      ++num_other;
    }
  }
  LOG(INFO) << "Bernoulli: zeros: "  << bernoulli_num_zeros
            << "; ones: " << num_ones << "; other: " << num_other;
  EXPECT_EQ(0, num_other);
  EXPECT_EQ(sample_size * empirical_mean, num_ones);
  EXPECT_EQ(sample_size * (1.0 - empirical_mean), bernoulli_num_zeros);
  // Multiply Uniform by Bernoulli
  for (int i = 0; i < sample_size; ++i) {
    samples[i] *= bernoulli_samples[i];
  }
  num_pos = 0;
  num_neg = 0;
  num_zeros = 0;
  for (int i = 0; i < sample_size; ++i) {
    if (samples[i] == TypeParam(0)) {
      ++num_zeros;
    } else if (samples[i] > TypeParam(0)) {
      ++num_pos;
    } else if (samples[i] < TypeParam(0)) {
      ++num_neg;
    }
  }
  // Check that we have as many zeros as Bernoulli, and roughly half positives
  // and half negatives (with bound computed from a Bernoulli with p = 0.5).
  EXPECT_EQ(bernoulli_num_zeros, num_zeros);
  p = 0.5;
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  int sub_sample_size = sample_size - bernoulli_num_zeros;
  bound = this->mean_bound(true_std, sub_sample_size);
  expected_num_each_sign = sub_sample_size * p;
  LOG(INFO) << "Uniform*Bernoulli: Expected " << expected_num_each_sign
            << " positives; got " << num_pos;
  LOG(INFO) << "Uniform*Bernoulli: Expected " << expected_num_each_sign
            << " negatives; got " << num_neg;
  EXPECT_NEAR(expected_num_each_sign, num_pos, sample_size * bound);
  EXPECT_NEAR(expected_num_each_sign, num_neg, sample_size * bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulliTimesBernoulli) {
  size_t sample_size = 10000;
  SyncedMemory bernoulli1_data(sample_size * sizeof(int));
  SyncedMemory bernoulli2_data(sample_size * sizeof(int));
  Caffe::set_random_seed(1701);
  double p1 = 0.5;
  caffe_vRngBernoulli(sample_size, reinterpret_cast<int*>(
      bernoulli1_data.mutable_cpu_data()), p1);
  TypeParam empirical_mean = this->sample_mean(
      reinterpret_cast<const int*>(bernoulli1_data.cpu_data()),
      sample_size);
  int bernoulli1_num_zeros = 0;
  int num_ones = 0;
  int num_other = 0;
  int* bernoulli_samples =
      reinterpret_cast<int*>(bernoulli1_data.mutable_cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (bernoulli_samples[i] == 0) {
      ++bernoulli1_num_zeros;
    } else if (bernoulli_samples[i] == 1) {
      ++num_ones;
    } else {
      ++num_other;
    }
  }
  TypeParam true_mean = p1;
  TypeParam true_std = sqrt(p1 * (1 - p1));
  TypeParam bound = this->mean_bound(true_std, sample_size);
  TypeParam expected_num_zeros = sample_size * (1 - true_mean);
  TypeParam expected_num_ones = sample_size * true_mean;
  LOG(INFO) << "Bernoulli1: Expected mean = " << true_mean
            << "; sample mean = " << empirical_mean;
  LOG(INFO) << "Bernoulli1: zeros: "  << bernoulli1_num_zeros
            << "; ones: " << num_ones << "; other: " << num_other;
  empirical_mean =
      this->sample_mean((const int *)bernoulli2_data.cpu_data(), sample_size);
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  EXPECT_EQ(num_other, 0);
  // Sample from Bernoulli with p = 0.3
  double p = 0.3;
  caffe_vRngBernoulli(sample_size,
      reinterpret_cast<int*>(bernoulli2_data.mutable_cpu_data()), p);
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  bound = this->mean_bound(true_std, sample_size);
  empirical_mean =
      this->sample_mean((const int *)bernoulli2_data.cpu_data(), sample_size);
  LOG(INFO) << "Bernoulli2: Expected mean = " << true_mean
            << "; sample mean = " << empirical_mean;
  EXPECT_NEAR(empirical_mean, true_mean, bound);
  int bernoulli2_num_zeros = 0;
  num_ones = 0;
  num_other = 0;
  const int* bernoulli2_samples =
      reinterpret_cast<const int*>(bernoulli2_data.cpu_data());
  for (int i = 0; i < sample_size; ++i) {
    if (bernoulli2_samples[i] == 0) {
      ++bernoulli2_num_zeros;
    } else if (bernoulli2_samples[i] == 1) {
      ++num_ones;
    } else {
      ++num_other;
    }
  }
  LOG(INFO) << "Bernoulli2: zeros: "  << bernoulli2_num_zeros
            << "; ones: " << num_ones << "; other: " << num_other;
  EXPECT_EQ(0, num_other);
  EXPECT_EQ(sample_size * empirical_mean, num_ones);
  EXPECT_EQ(sample_size * (1.0 - empirical_mean), bernoulli2_num_zeros);
  // Multiply Bernoulli1 by Bernoulli2
  for (int i = 0; i < sample_size; ++i) {
    bernoulli_samples[i] *= bernoulli2_samples[i];
  }
  bernoulli1_num_zeros = 0;
  num_ones = 0;
  num_other = 0;
  for (int i = 0; i < sample_size; ++i) {
    if (bernoulli_samples[i] == 0) {
      ++bernoulli1_num_zeros;
    } else if (bernoulli_samples[i] == 1) {
      ++num_ones;
    } else {
      ++num_other;
    }
  }
  // Check that we have as many zeros as Bernoulli, and roughly half positives
  // and half negatives (with bound computed from a Bernoulli with p = 0.5).
  p *= p1;
  true_mean = p;
  true_std = sqrt(p * (1 - p));
  empirical_mean =
      this->sample_mean((const int *)bernoulli2_data.cpu_data(), sample_size);
  bound = this->mean_bound(true_std, sample_size);
  LOG(INFO) << "Bernoulli1*Bernoulli2: Expected mean = " << true_mean
            << "; sample mean = " << empirical_mean;
  EXPECT_NEAR(empirical_mean, true_mean, bound);
}


}  // namespace caffe
