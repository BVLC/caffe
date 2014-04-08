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
 protected:
  RandomNumberGeneratorTest()
     : sample_size_(10000),
       seed_(1701),
       data_(new SyncedMemory(sample_size_ * sizeof(Dtype))),
       int_data_(new SyncedMemory(sample_size_ * sizeof(int))),
       int_data_2_(new SyncedMemory(sample_size_ * sizeof(int))) {}

  virtual void SetUp() {
    Caffe::set_random_seed(this->seed_);
  }

  Dtype sample_mean(const Dtype* const seqs, const int sample_size) {
    Dtype sum = 0;
    for (int i = 0; i < sample_size; ++i) {
      sum += seqs[i];
    }
    return sum / sample_size;
  }

  Dtype sample_mean(const Dtype* const seqs) {
    return sample_mean(seqs, sample_size_);
  }

  Dtype sample_mean(const int* const seqs, const int sample_size) {
    Dtype sum = 0;
    for (int i = 0; i < sample_size; ++i) {
      sum += Dtype(seqs[i]);
    }
    return sum / sample_size;
  }

  Dtype sample_mean(const int* const seqs) {
    return sample_mean(seqs, sample_size_);
  }

  Dtype mean_bound(const Dtype std, const int sample_size) {
    return std / sqrt(static_cast<Dtype>(sample_size));
  }

  Dtype mean_bound(const Dtype std) {
    return mean_bound(std, sample_size_);
  }

  void RngGaussianTest(const Dtype mu, const Dtype sigma, void* cpu_data) {
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_gaussian(sample_size_, mu, sigma, rng_data);
    const Dtype true_mean = mu;
    const Dtype true_std = sigma;
    // Check that sample mean roughly matches true mean.
    const Dtype bound = this->mean_bound(true_std);
    const Dtype sample_mean = this->sample_mean(
        static_cast<const Dtype*>(cpu_data));
    EXPECT_NEAR(sample_mean, true_mean, bound);
    // Check that roughly half the samples are above the true mean.
    int num_above_mean = 0;
    int num_below_mean = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (rng_data[i] > true_mean) {
        ++num_above_mean;
      } else if (rng_data[i] < true_mean) {
        ++num_below_mean;
      }
    }
    EXPECT_EQ(sample_size_, num_above_mean + num_below_mean);
    const Dtype sample_p_above_mean =
        static_cast<Dtype>(num_above_mean) / sample_size_;
    const Dtype bernoulli_p = 0.5;
    const Dtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Dtype bernoulli_bound = this->mean_bound(true_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, bernoulli_bound);
  }

  void RngUniformTest(const Dtype lower, const Dtype upper, void* cpu_data) {
    CHECK_GE(upper, lower);
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_uniform(sample_size_, lower, upper, rng_data);
    const Dtype true_mean = (lower + upper) / 2;
    const Dtype true_std = (upper - lower) / sqrt(12);
    // Check that sample mean roughly matches true mean.
    const Dtype bound = this->mean_bound(true_std);
    const Dtype sample_mean = this->sample_mean(rng_data);
    EXPECT_NEAR(sample_mean, true_mean, bound);
    // Check that roughly half the samples are above the true mean, and none are
    // above upper or below lower.
    int num_above_mean = 0;
    int num_below_mean = 0;
    int num_above_upper = 0;
    int num_below_lower = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (rng_data[i] > true_mean) {
        ++num_above_mean;
      } else if (rng_data[i] < true_mean) {
        ++num_below_mean;
      }
      if (rng_data[i] > upper) {
        ++num_above_upper;
      } else if (rng_data[i] < lower) {
        ++num_below_lower;
      }
    }
    EXPECT_EQ(0, num_above_upper);
    EXPECT_EQ(0, num_below_lower);
    EXPECT_EQ(sample_size_, num_above_mean + num_below_mean);
    const Dtype sample_p_above_mean =
        static_cast<Dtype>(num_above_mean) / sample_size_;
    const Dtype bernoulli_p = 0.5;
    const Dtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Dtype bernoulli_bound = this->mean_bound(true_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, bernoulli_bound);
  }

  void RngBernoulliTest(const Dtype p, void* cpu_data) {
    int* rng_data = static_cast<int*>(cpu_data);
    caffe_rng_bernoulli(sample_size_, p, rng_data);
    const Dtype true_mean = p;
    const Dtype true_std = sqrt(p * (1 - p));
    const Dtype bound = this->mean_bound(true_std);
    const Dtype sample_mean = this->sample_mean(rng_data);
    EXPECT_NEAR(sample_mean, true_mean, bound);
  }

  int num_above_mean;
  int num_below_mean;

  size_t sample_size_;
  uint32_t seed_;

  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> int_data_;
  shared_ptr<SyncedMemory> int_data_2_;
};


typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(RandomNumberGeneratorTest, Dtypes);


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian) {
  const TypeParam mu = 0;
  const TypeParam sigma = 1;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianTest(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian2) {
  const TypeParam mu = -2;
  const TypeParam sigma = 3;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianTest(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform) {
  const TypeParam lower = 0;
  const TypeParam upper = 1;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformTest(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform2) {
  const TypeParam lower = -7.3;
  const TypeParam upper = -2.3;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformTest(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli) {
  const TypeParam p = 0.3;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliTest(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli2) {
  const TypeParam p = 0.9;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliTest(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesBernoulli) {
  // Sample from 0 mean Gaussian.
  const TypeParam mu = 0;
  const TypeParam sigma = 1;
  TypeParam* gaussian_data =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngGaussianTest(mu, sigma, gaussian_data);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliTest(bernoulli_p, bernoulli_data);

  // Multiply Gaussian by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data[i] *= bernoulli_data[i];
  }
  int num_pos = 0;
  int num_neg = 0;
  for (int i = 0; i < this->sample_size_; ++i) {
    if (gaussian_data[i] == TypeParam(0)) {
      EXPECT_EQ(TypeParam(0), bernoulli_data[i]);
    } else {
      EXPECT_EQ(TypeParam(1), bernoulli_data[i]);
      if (gaussian_data[i] > TypeParam(0)) {
        ++num_pos;
      } else if (gaussian_data[i] < TypeParam(0)) {
        ++num_neg;
      }
    }
  }

  // Check that Gaussian still has roughly half positives and half negatives
  // (with bound computed from a Bernoulli with p = 0.5).
  const int num_non_zero = num_pos + num_neg;
  const TypeParam sample_p = num_pos / static_cast<TypeParam>(num_non_zero);
  const TypeParam p = 0.5;
  const TypeParam true_mean = p;
  const TypeParam true_std = sqrt(p * (1 - p));
  const TypeParam bound = this->mean_bound(true_std, num_non_zero);
  EXPECT_NEAR(true_mean, sample_p, bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesBernoulli) {
  // Sample from Uniform on [-1, 1].
  const TypeParam lower = -1;
  const TypeParam upper = 1;
  TypeParam* uniform_data =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngUniformTest(lower, upper, uniform_data);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliTest(bernoulli_p, bernoulli_data);

  // Multiply Uniform by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data[i] *= bernoulli_data[i];
  }
  int num_pos = 0;
  int num_neg = 0;
  for (int i = 0; i < this->sample_size_; ++i) {
    if (uniform_data[i] == TypeParam(0)) {
      EXPECT_EQ(TypeParam(0), bernoulli_data[i]);
    } else {
      EXPECT_EQ(TypeParam(1), bernoulli_data[i]);
      if (uniform_data[i] > TypeParam(0)) {
        ++num_pos;
      } else if (uniform_data[i] < TypeParam(0)) {
        ++num_neg;
      }
    }
  }

  // Check that Uniform still has roughly half positives and half negatives
  // (with bound computed from a Bernoulli with p = 0.5).
  const int num_non_zero = num_pos + num_neg;
  const TypeParam sample_p = num_pos / static_cast<TypeParam>(num_non_zero);
  const TypeParam p = 0.5;
  const TypeParam true_mean = p;
  const TypeParam true_std = sqrt(p * (1 - p));
  const TypeParam bound = this->mean_bound(true_std, num_non_zero);
  EXPECT_NEAR(true_mean, sample_p, bound);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulliTimesBernoulli) {
  // Sample from Bernoulli with p = 0.5.
  const TypeParam p_a = 0.5;
  int* bernoulli_data_a =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliTest(p_a, bernoulli_data_a);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam p_b = 0.3;
  int* bernoulli_data_b =
      static_cast<int*>(this->int_data_2_->mutable_cpu_data());
  this->RngBernoulliTest(p_b, bernoulli_data_b);

  // Multiply Bernoullis.
  for (int i = 0; i < this->sample_size_; ++i) {
    bernoulli_data_a[i] *= bernoulli_data_b[i];
  }
  int num_ones = 0;
  for (int i = 0; i < this->sample_size_; ++i) {
    if (bernoulli_data_a[i] != TypeParam(0)) {
      EXPECT_EQ(TypeParam(1), bernoulli_data_a[i]);
      ++num_ones;
    }
  }

  // Check that resulting product has roughly p_a * p_b ones.
  const TypeParam sample_p = this->sample_mean(bernoulli_data_a);
  const TypeParam true_mean = p_a * p_b;
  const TypeParam true_std = sqrt(true_mean * (1 - true_mean));
  const TypeParam bound = this->mean_bound(true_std);
  EXPECT_NEAR(true_mean, sample_p, bound);
}


}  // namespace caffe
