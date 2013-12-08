// Copyright 2013 Yangqing Jia

#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

class CommonTest : public ::testing::Test {};

TEST_F(CommonTest, TestCublasHandler) {
  int cuda_device_id;
  CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  EXPECT_TRUE(Caffe::cublas_handle());
}

TEST_F(CommonTest, TestVslStream) {
  //EXPECT_TRUE(Caffe::vsl_stream());
    EXPECT_TRUE(true);
}

TEST_F(CommonTest, TestBrewMode) {
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

TEST_F(CommonTest, TestPhase) {
  EXPECT_EQ(Caffe::phase(), Caffe::TRAIN);
  Caffe::set_phase(Caffe::TEST);
  EXPECT_EQ(Caffe::phase(), Caffe::TEST);
}

TEST_F(CommonTest, TestRandSeedCPU) {
  SyncedMemory data_a(10 * sizeof(int));
  SyncedMemory data_b(10 * sizeof(int));
  Caffe::set_random_seed(1701);
  //viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, Caffe::vsl_stream(),
  //      10, (int*)data_a.mutable_cpu_data(), 0.5);
  caffe_vRngBernoulli(10, (int*)data_a.mutable_cpu_data(), 0.5);

  Caffe::set_random_seed(1701);
  //viRngBernoulli(VSL_RNG_METHOD_BERNOULLI_ICDF, Caffe::vsl_stream(),
  //      10, (int*)data_b.mutable_cpu_data(), 0.5);
  caffe_vRngBernoulli(10, (int*)data_b.mutable_cpu_data(), 0.5);

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const int*)(data_a.cpu_data()))[i],
        ((const int*)(data_b.cpu_data()))[i]);
  }
}


TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        (unsigned int*)data_a.mutable_gpu_data(), 10));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        (unsigned int*)data_b.mutable_gpu_data(), 10));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}


}  // namespace caffe
