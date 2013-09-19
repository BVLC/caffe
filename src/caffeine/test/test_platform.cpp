#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "caffeine/test/test_caffeine_main.hpp"

namespace caffeine {

extern cudaDeviceProp CAFFEINE_TEST_CUDA_PROP;

class PlatformTest : public ::testing::Test {};

TEST_F(PlatformTest, TestInitialization) {
  printf("Major revision number:         %d\n",  CAFFEINE_TEST_CUDA_PROP.major);
  printf("Minor revision number:         %d\n",  CAFFEINE_TEST_CUDA_PROP.minor);
  printf("Name:                          %s\n",  CAFFEINE_TEST_CUDA_PROP.name);
  printf("Total global memory:           %lu\n",  CAFFEINE_TEST_CUDA_PROP.totalGlobalMem);
  printf("Total shared memory per block: %lu\n",  CAFFEINE_TEST_CUDA_PROP.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  CAFFEINE_TEST_CUDA_PROP.regsPerBlock);
  printf("Warp size:                     %d\n",  CAFFEINE_TEST_CUDA_PROP.warpSize);
  printf("Maximum memory pitch:          %lu\n",  CAFFEINE_TEST_CUDA_PROP.memPitch);
  printf("Maximum threads per block:     %d\n",  CAFFEINE_TEST_CUDA_PROP.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, CAFFEINE_TEST_CUDA_PROP.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, CAFFEINE_TEST_CUDA_PROP.maxGridSize[i]);
  printf("Clock rate:                    %d\n",  CAFFEINE_TEST_CUDA_PROP.clockRate);
  printf("Total constant memory:         %lu\n",  CAFFEINE_TEST_CUDA_PROP.totalConstMem);
  printf("Texture alignment:             %lu\n",  CAFFEINE_TEST_CUDA_PROP.textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (CAFFEINE_TEST_CUDA_PROP.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  CAFFEINE_TEST_CUDA_PROP.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",  (CAFFEINE_TEST_CUDA_PROP.kernelExecTimeoutEnabled ? "Yes" : "No"));
  EXPECT_TRUE(true);
}

}  // namespace caffeine
