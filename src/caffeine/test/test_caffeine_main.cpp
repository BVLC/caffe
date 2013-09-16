#include <iostream>

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

using namespace std;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::google::InitGoogleLogging(argv[0]);
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("Major revision number:         %d\n",  prop.major);
  printf("Minor revision number:         %d\n",  prop.minor);
  printf("Name:                          %s\n",  prop.name);
  printf("Total global memory:           %u\n",  prop.totalGlobalMem);
  printf("Total shared memory per block: %u\n",  prop.sharedMemPerBlock);
  printf("Total registers per block:     %d\n",  prop.regsPerBlock);
  printf("Warp size:                     %d\n",  prop.warpSize);
  printf("Maximum memory pitch:          %u\n",  prop.memPitch);
  printf("Maximum threads per block:     %d\n",  prop.maxThreadsPerBlock);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, prop.maxThreadsDim[i]);
  for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, prop.maxGridSize[i]);
  printf("Clock rate:                    %d\n",  prop.clockRate);
  printf("Total constant memory:         %u\n",  prop.totalConstMem);
  printf("Texture alignment:             %u\n",  prop.textureAlignment);
  printf("Concurrent copy and execution: %s\n",  (prop.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n",  prop.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",  (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
  
  return RUN_ALL_TESTS();
}
