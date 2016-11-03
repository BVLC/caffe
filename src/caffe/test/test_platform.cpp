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

#ifndef CPU_ONLY

#include <cstdio>
#include <cstdlib>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

class PlatformTest : public ::testing::Test {};

TEST_F(PlatformTest, TestInitialization) {
  printf("Major revision number:         %d\n",  CAFFE_TEST_CUDA_PROP.major);
  printf("Minor revision number:         %d\n",  CAFFE_TEST_CUDA_PROP.minor);
  printf("Name:                          %s\n",  CAFFE_TEST_CUDA_PROP.name);
  printf("Total global memory:           %lu\n",
         CAFFE_TEST_CUDA_PROP.totalGlobalMem);
  printf("Total shared memory per block: %lu\n",
         CAFFE_TEST_CUDA_PROP.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",
         CAFFE_TEST_CUDA_PROP.regsPerBlock);
  printf("Warp size:                     %d\n",
         CAFFE_TEST_CUDA_PROP.warpSize);
  printf("Maximum memory pitch:          %lu\n",
         CAFFE_TEST_CUDA_PROP.memPitch);
  printf("Maximum threads per block:     %d\n",
         CAFFE_TEST_CUDA_PROP.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i,
           CAFFE_TEST_CUDA_PROP.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i,
           CAFFE_TEST_CUDA_PROP.maxGridSize[i]);
  printf("Clock rate:                    %d\n", CAFFE_TEST_CUDA_PROP.clockRate);
  printf("Total constant memory:         %lu\n",
         CAFFE_TEST_CUDA_PROP.totalConstMem);
  printf("Texture alignment:             %lu\n",
         CAFFE_TEST_CUDA_PROP.textureAlignment);
  printf("Concurrent copy and execution: %s\n",
         (CAFFE_TEST_CUDA_PROP.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",
         CAFFE_TEST_CUDA_PROP.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",
         (CAFFE_TEST_CUDA_PROP.kernelExecTimeoutEnabled ? "Yes" : "No"));
  printf("Unified virtual addressing:    %s\n",
         (CAFFE_TEST_CUDA_PROP.unifiedAddressing ? "Yes" : "No"));
  EXPECT_TRUE(true);
}

}  // namespace caffe

#endif  // CPU_ONLY
