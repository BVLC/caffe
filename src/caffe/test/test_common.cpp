#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class CommonTest : public ::testing::Test {};

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestCublasHandlerGPU) {
  if (Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    int cuda_device_id;
    CUDA_CHECK(cudaGetDevice(&cuda_device_id));
    EXPECT_TRUE(Caffe::cublas_handle());
#endif  // USE_CUDA
  }
}

#endif

TEST_F(CommonTest, TestBrewMode) {
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

TEST_F(CommonTest, TestRandSeedCPU) {
  SyncedMemory data_a(10 * sizeof(int), Caffe::GetDefaultDevice());
  SyncedMemory data_b(10 * sizeof(int), Caffe::GetDefaultDevice());
  Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_a.mutable_cpu_data()));

  Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
  caffe_rng_bernoulli(10, 0.5, static_cast<int*>(data_b.mutable_cpu_data()));

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(static_cast<const int*>(data_a.cpu_data())[i],
        static_cast<const int*>(data_b.cpu_data())[i]);
  }
}

#ifndef CPU_ONLY  // GPU Caffe singleton test.

TEST_F(CommonTest, TestRandSeedGPU) {
  device *dc = Caffe::GetDefaultDevice();

  if (dc->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    SyncedMemory data_a(10 * sizeof(unsigned int),
                        Caffe::GetDefaultDevice());
    SyncedMemory data_b(10 * sizeof(unsigned int),
                        Caffe::GetDefaultDevice());
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
          static_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
    Caffe::set_random_seed(1701, Caffe::GetDefaultDevice());
    CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
          static_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
    for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
          ((const unsigned int*)(data_b.cpu_data()))[i]);
    }
#endif  // USE_CUDA
  }
}

#endif

}  // namespace caffe
