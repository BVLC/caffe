// Copyright 2014 BVLC and contributors.

#include <cstring>

#include "cuda_runtime.h"
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

TEST_F(CommonTest, TestBrewMode) {
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
}

TEST_F(CommonTest, TestPhase) {
  Caffe::set_phase(Caffe::TRAIN);
  EXPECT_EQ(Caffe::phase(), Caffe::TRAIN);
  Caffe::set_phase(Caffe::TEST);
  EXPECT_EQ(Caffe::phase(), Caffe::TEST);
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

TEST_F(CommonTest, TestRandSeedGPU) {
  SyncedMemory data_a(10 * sizeof(unsigned int));
  SyncedMemory data_b(10 * sizeof(unsigned int));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        reinterpret_cast<unsigned int*>(data_a.mutable_gpu_data()), 10));
  Caffe::set_random_seed(1701);
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(),
        reinterpret_cast<unsigned int*>(data_b.mutable_gpu_data()), 10));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(((const unsigned int*)(data_a.cpu_data()))[i],
        ((const unsigned int*)(data_b.cpu_data()))[i]);
  }
}

}  // namespace caffe
