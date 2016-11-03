/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

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
  Caffe::set_mode(Caffe::CPU);
  EXPECT_EQ(Caffe::mode(), Caffe::CPU);
  Caffe::set_mode(Caffe::GPU);
  EXPECT_EQ(Caffe::mode(), Caffe::GPU);
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

#endif

}  // namespace caffe
