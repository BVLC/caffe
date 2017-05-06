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
