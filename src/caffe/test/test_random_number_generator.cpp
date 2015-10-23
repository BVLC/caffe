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
     : mean_bound_multiplier_(3.8),  // ~99.99% confidence for test failure.
       sample_size_(10000),
       seed_(1701),
       data_(new SyncedMemory(sample_size_ * sizeof(Dtype))),
       data_2_(new SyncedMemory(sample_size_ * sizeof(Dtype))),
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
    return mean_bound_multiplier_ * std / sqrt(static_cast<Dtype>(sample_size));
  }

  Dtype mean_bound(const Dtype std) {
    return mean_bound(std, sample_size_);
  }

  void RngGaussianFill(const Dtype mu, const Dtype sigma, void* cpu_data) {
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_gaussian(sample_size_, mu, sigma, rng_data);
  }

  void RngGaussianChecks(const Dtype mu, const Dtype sigma,
                         const void* cpu_data, const Dtype sparse_p = 0) {
    const Dtype* rng_data = static_cast<const Dtype*>(cpu_data);
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
    int num_mean = 0;
    int num_nan = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (rng_data[i] > true_mean) {
        ++num_above_mean;
      } else if (rng_data[i] < true_mean) {
        ++num_below_mean;
      } else if (rng_data[i] == true_mean) {
        ++num_mean;
      } else {
        ++num_nan;
      }
    }
    EXPECT_EQ(0, num_nan);
    if (sparse_p == Dtype(0)) {
      EXPECT_EQ(0, num_mean);
    }
    const Dtype sample_p_above_mean =
        static_cast<Dtype>(num_above_mean) / sample_size_;
    const Dtype bernoulli_p = (1 - sparse_p) * 0.5;
    const Dtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Dtype bernoulli_bound = this->mean_bound(bernoulli_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, bernoulli_bound);
  }

  void RngUniformFill(const Dtype lower, const Dtype upper, void* cpu_data) {
    CHECK_GE(upper, lower);
    Dtype* rng_data = static_cast<Dtype*>(cpu_data);
    caffe_rng_uniform(sample_size_, lower, upper, rng_data);
  }

  void RngUniformChecks(const Dtype lower, const Dtype upper,
                        const void* cpu_data, const Dtype sparse_p = 0) {
    const Dtype* rng_data = static_cast<const Dtype*>(cpu_data);
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
    int num_mean = 0;
    int num_nan = 0;
    int num_above_upper = 0;
    int num_below_lower = 0;
    for (int i = 0; i < sample_size_; ++i) {
      if (rng_data[i] > true_mean) {
        ++num_above_mean;
      } else if (rng_data[i] < true_mean) {
        ++num_below_mean;
      } else if (rng_data[i] == true_mean) {
        ++num_mean;
      } else {
        ++num_nan;
      }
      if (rng_data[i] > upper) {
        ++num_above_upper;
      } else if (rng_data[i] < lower) {
        ++num_below_lower;
      }
    }
    EXPECT_EQ(0, num_nan);
    EXPECT_EQ(0, num_above_upper);
    EXPECT_EQ(0, num_below_lower);
    if (sparse_p == Dtype(0)) {
      EXPECT_EQ(0, num_mean);
    }
    const Dtype sample_p_above_mean =
        static_cast<Dtype>(num_above_mean) / sample_size_;
    const Dtype bernoulli_p = (1 - sparse_p) * 0.5;
    const Dtype bernoulli_std = sqrt(bernoulli_p * (1 - bernoulli_p));
    const Dtype bernoulli_bound = this->mean_bound(bernoulli_std);
    EXPECT_NEAR(bernoulli_p, sample_p_above_mean, bernoulli_bound);
  }

  void RngBernoulliFill(const Dtype p, void* cpu_data) {
    int* rng_data = static_cast<int*>(cpu_data);
    caffe_rng_bernoulli(sample_size_, p, rng_data);
  }

  void RngBernoulliChecks(const Dtype p, const void* cpu_data) {
    const int* rng_data = static_cast<const int*>(cpu_data);
    const Dtype true_mean = p;
    const Dtype true_std = sqrt(p * (1 - p));
    const Dtype bound = this->mean_bound(true_std);
    const Dtype sample_mean = this->sample_mean(rng_data);
    EXPECT_NEAR(sample_mean, true_mean, bound);
  }

#ifndef CPU_ONLY

  void RngGaussianFillGPU(const Dtype mu, const Dtype sigma, void* gpu_data) {
    Dtype* rng_data = static_cast<Dtype*>(gpu_data);
    caffe_gpu_rng_gaussian(sample_size_, mu, sigma, rng_data);
  }

  void RngUniformFillGPU(const Dtype lower, const Dtype upper, void* gpu_data) {
    CHECK_GE(upper, lower);
    Dtype* rng_data = static_cast<Dtype*>(gpu_data);
    caffe_gpu_rng_uniform(sample_size_, lower, upper, rng_data);
  }

  // Fills with uniform integers in [0, UINT_MAX] using 2 argument form of
  // caffe_gpu_rng_uniform.
  void RngUniformIntFillGPU(void* gpu_data) {
    unsigned int* rng_data = static_cast<unsigned int*>(gpu_data);
    caffe_gpu_rng_uniform(sample_size_, rng_data);
  }

#endif

  int num_above_mean;
  int num_below_mean;

  Dtype mean_bound_multiplier_;

  size_t sample_size_;
  uint32_t seed_;

  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> data_2_;
  shared_ptr<SyncedMemory> int_data_;
  shared_ptr<SyncedMemory> int_data_2_;
};

TYPED_TEST_CASE(RandomNumberGeneratorTest, TestDtypes);

TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian) {
  const TypeParam mu = 0;
  const TypeParam sigma = 1;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianFill(mu, sigma, gaussian_data);
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian2) {
  const TypeParam mu = -2;
  const TypeParam sigma = 3;
  void* gaussian_data = this->data_->mutable_cpu_data();
  this->RngGaussianFill(mu, sigma, gaussian_data);
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform) {
  const TypeParam lower = 0;
  const TypeParam upper = 1;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformFill(lower, upper, uniform_data);
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform2) {
  const TypeParam lower = -7.3;
  const TypeParam upper = -2.3;
  void* uniform_data = this->data_->mutable_cpu_data();
  this->RngUniformFill(lower, upper, uniform_data);
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli) {
  const TypeParam p = 0.3;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliFill(p, bernoulli_data);
  this->RngBernoulliChecks(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulli2) {
  const TypeParam p = 0.9;
  void* bernoulli_data = this->int_data_->mutable_cpu_data();
  this->RngBernoulliFill(p, bernoulli_data);
  this->RngBernoulliChecks(p, bernoulli_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesGaussian) {
  const TypeParam mu = 0;
  const TypeParam sigma = 1;

  // Sample from 0 mean Gaussian.
  TypeParam* gaussian_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data_1);

  // Sample from 0 mean Gaussian again.
  TypeParam* gaussian_data_2 =
      static_cast<TypeParam*>(this->data_2_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data_2);

  // Multiply Gaussians.
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data_1[i] *= gaussian_data_2[i];
  }

  // Check that result has mean 0.
  TypeParam mu_product = pow(mu, 2);
  TypeParam sigma_product = sqrt(pow(sigma, 2) / 2);
  this->RngGaussianChecks(mu_product, sigma_product, gaussian_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesUniform) {
  // Sample from Uniform on [-2, 2].
  const TypeParam lower_1 = -2;
  const TypeParam upper_1 = -lower_1;
  TypeParam* uniform_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngUniformFill(lower_1, upper_1, uniform_data_1);

  // Sample from Uniform on [-3, 3].
  const TypeParam lower_2 = -3;
  const TypeParam upper_2 = -lower_2;
  TypeParam* uniform_data_2 =
      static_cast<TypeParam*>(this->data_2_->mutable_cpu_data());
  this->RngUniformFill(lower_2, upper_2, uniform_data_2);

  // Multiply Uniforms.
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data_1[i] *= uniform_data_2[i];
  }

  // Check that result does not violate checked properties of Uniform on [-6, 6]
  // (though it is not actually uniformly distributed).
  const TypeParam lower_prod = lower_1 * upper_2;
  const TypeParam upper_prod = -lower_prod;
  this->RngUniformChecks(lower_prod, upper_prod, uniform_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesBernoulli) {
  // Sample from 0 mean Gaussian.
  const TypeParam mu = 0;
  const TypeParam sigma = 1;
  TypeParam* gaussian_data =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngGaussianFill(mu, sigma, gaussian_data);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(bernoulli_p, bernoulli_data);

  // Multiply Gaussian by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data[i] *= bernoulli_data[i];
  }

  // Check that result does not violate checked properties of sparsified
  // Gaussian (though it is not actually a Gaussian).
  this->RngGaussianChecks(mu, sigma, gaussian_data, 1 - bernoulli_p);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesBernoulli) {
  // Sample from Uniform on [-1, 1].
  const TypeParam lower = -1;
  const TypeParam upper = 1;
  TypeParam* uniform_data =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  this->RngUniformFill(lower, upper, uniform_data);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam bernoulli_p = 0.3;
  int* bernoulli_data =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(bernoulli_p, bernoulli_data);

  // Multiply Uniform by Bernoulli.
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data[i] *= bernoulli_data[i];
  }

  // Check that result does not violate checked properties of sparsified
  // Uniform on [-1, 1] (though it is not actually uniformly distributed).
  this->RngUniformChecks(lower, upper, uniform_data, 1 - bernoulli_p);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngBernoulliTimesBernoulli) {
  // Sample from Bernoulli with p = 0.5.
  const TypeParam p_a = 0.5;
  int* bernoulli_data_a =
      static_cast<int*>(this->int_data_->mutable_cpu_data());
  this->RngBernoulliFill(p_a, bernoulli_data_a);

  // Sample from Bernoulli with p = 0.3.
  const TypeParam p_b = 0.3;
  int* bernoulli_data_b =
      static_cast<int*>(this->int_data_2_->mutable_cpu_data());
  this->RngBernoulliFill(p_b, bernoulli_data_b);

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

#ifndef CPU_ONLY

TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianGPU) {
  const TypeParam mu = 0;
  const TypeParam sigma = 1;
  void* gaussian_gpu_data = this->data_->mutable_gpu_data();
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data);
  const void* gaussian_data = this->data_->cpu_data();
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussian2GPU) {
  const TypeParam mu = -2;
  const TypeParam sigma = 3;
  void* gaussian_gpu_data = this->data_->mutable_gpu_data();
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data);
  const void* gaussian_data = this->data_->cpu_data();
  this->RngGaussianChecks(mu, sigma, gaussian_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformGPU) {
  const TypeParam lower = 0;
  const TypeParam upper = 1;
  void* uniform_gpu_data = this->data_->mutable_gpu_data();
  this->RngUniformFillGPU(lower, upper, uniform_gpu_data);
  const void* uniform_data = this->data_->cpu_data();
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniform2GPU) {
  const TypeParam lower = -7.3;
  const TypeParam upper = -2.3;
  void* uniform_gpu_data = this->data_->mutable_gpu_data();
  this->RngUniformFillGPU(lower, upper, uniform_gpu_data);
  const void* uniform_data = this->data_->cpu_data();
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformIntGPU) {
  unsigned int* uniform_uint_gpu_data =
      static_cast<unsigned int*>(this->int_data_->mutable_gpu_data());
  this->RngUniformIntFillGPU(uniform_uint_gpu_data);
  const unsigned int* uniform_uint_data =
      static_cast<const unsigned int*>(this->int_data_->cpu_data());
  TypeParam* uniform_data =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data[i] = static_cast<const TypeParam>(uniform_uint_data[i]);
  }
  const TypeParam lower = 0;
  const TypeParam upper = UINT_MAX;
  this->RngUniformChecks(lower, upper, uniform_data);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngGaussianTimesGaussianGPU) {
  const TypeParam mu = 0;
  const TypeParam sigma = 1;

  // Sample from 0 mean Gaussian.
  TypeParam* gaussian_gpu_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_gpu_data());
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data_1);

  // Sample from 0 mean Gaussian again.
  TypeParam* gaussian_gpu_data_2 =
      static_cast<TypeParam*>(this->data_2_->mutable_gpu_data());
  this->RngGaussianFillGPU(mu, sigma, gaussian_gpu_data_2);

  // Multiply Gaussians.
  TypeParam* gaussian_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  const TypeParam* gaussian_data_2 =
      static_cast<const TypeParam*>(this->data_2_->cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    gaussian_data_1[i] *= gaussian_data_2[i];
  }

  // Check that result does not violate checked properties of Gaussian
  // (though it is not actually a Gaussian).
  TypeParam mu_product = pow(mu, 2);
  TypeParam sigma_product = sqrt(pow(sigma, 2) / 2);
  this->RngGaussianChecks(mu_product, sigma_product, gaussian_data_1);
}


TYPED_TEST(RandomNumberGeneratorTest, TestRngUniformTimesUniformGPU) {
  // Sample from Uniform on [-2, 2].
  const TypeParam lower_1 = -2;
  const TypeParam upper_1 = -lower_1;
  TypeParam* uniform_gpu_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_gpu_data());
  this->RngUniformFillGPU(lower_1, upper_1, uniform_gpu_data_1);

  // Sample from Uniform on [-3, 3].
  const TypeParam lower_2 = -3;
  const TypeParam upper_2 = -lower_2;
  TypeParam* uniform_gpu_data_2 =
      static_cast<TypeParam*>(this->data_2_->mutable_gpu_data());
  this->RngUniformFillGPU(lower_2, upper_2, uniform_gpu_data_2);

  // Multiply Uniforms.
  TypeParam* uniform_data_1 =
      static_cast<TypeParam*>(this->data_->mutable_cpu_data());
  const TypeParam* uniform_data_2 =
      static_cast<const TypeParam*>(this->data_2_->cpu_data());
  for (int i = 0; i < this->sample_size_; ++i) {
    uniform_data_1[i] *= uniform_data_2[i];
  }

  // Check that result does not violate properties of Uniform on [-7, -3].
  const TypeParam lower_prod = lower_1 * upper_2;
  const TypeParam upper_prod = -lower_prod;
  this->RngUniformChecks(lower_prod, upper_prod, uniform_data_1);
}

#endif

}  // namespace caffe
