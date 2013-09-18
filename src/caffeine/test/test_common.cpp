#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffeine/common.hpp"
#include "caffeine/syncedmem.hpp"

namespace caffeine {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, TestCublasHandler) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffeine::cublas_handle());
}

TEST_F(CommonTest, TestVslStream) {
  EXPECT_TRUE(Caffeine::vsl_stream());
}

TEST_F(CommonTest, TestBrewMode) {
  EXPECT_EQ(Caffeine::mode(), Caffeine::CPU);
  Caffeine::set_mode(Caffeine::GPU);
  EXPECT_EQ(Caffeine::mode(), Caffeine::GPU);
}

TEST_F(CommonTest, TestPhase) {
  EXPECT_EQ(Caffeine::phase(), Caffeine::TRAIN);
  Caffeine::set_phase(Caffeine::TEST);
  EXPECT_EQ(Caffeine::phase(), Caffeine::TEST);
}

TEST_F(CommonTest, TestRandSeedCPU) {
  SyncedMemory data_a(10 * sizeof(int));
  SyncedMemory data_b(10 * sizeof(int));
  Caffeine::set_random_seed(1701);
  viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, Caffeine::vsl_stream(),
        10, (int*)data_a.mutable_cpu_data(), 0.5);
  Caffeine::set_random_seed(1701);
  viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, Caffeine::vsl_stream(),
        10, (int*)data_b.mutable_cpu_data(), 0.5);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const int*)(data_a.cpu_data()))[i],
        ((const int*)(data_b.cpu_data()))[i]);
  }
}


TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffeine::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffeine::curand_generator(),
        (unsigned int*)data_a.mutable_gpu_data(), 10));
  Caffeine::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffeine::curand_generator(),
        (unsigned int*)data_b.mutable_gpu_data(), 10));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}


}  // namespace caffeine
