#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

#ifndef CPU_ONLY
#include "cub/util_allocator.cuh"
#endif

namespace caffe {

class CommonTest : public ::testing::Test {};

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestCublasHandlerGPU) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffe::cublas_handle());
}

#endif

TEST_F(CommonTest, TestBrewMode) {
  Caffe::Brew current_mode = Caffe::mode();
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
  Caffe::set_mode(current_mode);
}

TEST_F(CommonTest, TestRandSeedCPU) {
  SyncedMemory data_a(10 * sizeof(int));
  SyncedMemory data_b(10 * sizeof(int));
  Caffe::set_random_seed(1701);
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_a.mutable_cpu_data()));

  Caffe::set_random_seed(1701);
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_b.mutable_cpu_data()));

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<const int*>(data_a.cpu_data())[i],
        static_cast<const int*>(data_b.cpu_data())[i]);
  }
}

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        static_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        static_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}

size_t pow2(unsigned int p) {
  return p > 0 ? (2ULL << (p-1)) : 1;
}

TEST_F(CommonTest, TestCUBNearestPowerOf2) {
  size_t rounded_bytes;
  unsigned int power;
  for (int p = 0; p < sizeof(size_t) * CHAR_BIT; ++p) {
    size_t value = pow2(p);
    ++value;
    cub::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    EXPECT_EQ(p + 1, power);
    EXPECT_EQ(pow2(power), rounded_bytes);
    --value;
    cub::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    EXPECT_EQ(p, power);
    EXPECT_EQ(pow2(power), rounded_bytes);
    --value;
    cub::CachingDeviceAllocator::NearestPowerOf(power, rounded_bytes, 2, value);
    // Exclusion: for zero size we return 1 as rounded bytes (per original CUB
    // design)
    EXPECT_EQ(p == 1 ? 0 : p, power);  // because 2^1 - 1 == 2^0
    EXPECT_EQ(pow2(power), rounded_bytes);
  }
}

#endif

}  // namespace caffe
