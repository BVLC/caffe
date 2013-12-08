// Copyright 2013 Yangqing Jia

#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;


long cluster_seedgen(void) {
  long s, seed, pid;
  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


Caffe::Caffe()
    : mode_(Caffe::CPU), phase_(Caffe::TRAIN), cublas_handle_(NULL),
      curand_generator_(NULL),
      //vsl_stream_(NULL)
      random_generator_()
{
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  }
  // Try to create a curand handler.
  if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
      != CURAND_STATUS_SUCCESS ||
      curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
      != CURAND_STATUS_SUCCESS) {
    LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  }

  // Try to create a vsl stream. This should almost always work, but we will
  // check it anyway.
  //if (vslNewStream(&vsl_stream_, VSL_BRNG_MT19937, cluster_seedgen()) != VSL_STATUS_OK) {
  //  LOG(ERROR) << "Cannot create vsl stream. VSL random number generator "
  //      << "won't be available.";
  //}
}

Caffe::~Caffe() {
  if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  if (curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  }
  //if (vsl_stream_) VSL_CHECK(vslDeleteStream(&vsl_stream_));
}

void Caffe::set_random_seed(const unsigned int seed) {
  // Curand seed
  // Yangqing's note: simply setting the generator seed does not seem to
  // work on the tesla K20s, so I wrote the ugly reset thing below.
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(curand_generator()));
    CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
        CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
        seed));
  } else {
    LOG(ERROR) << "Curand not available. Skipping setting the curand seed.";
  }
  // VSL seed
  //VSL_CHECK(vslDeleteStream(&(Get().vsl_stream_)));
  //VSL_CHECK(vslNewStream(&(Get().vsl_stream_), VSL_BRNG_MT19937, seed));
  Get().random_generator_ = random_generator_t(seed);

}

void Caffe::SetDevice(const int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  if (Get().curand_generator_) {
    CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  }
  CUDA_CHECK(cudaSetDevice(device_id));
  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
      CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
      cluster_seedgen()));
}

void Caffe::DeviceQuery() {
  cudaDeviceProp prop;
  int device;
  if (cudaSuccess != cudaGetDevice(&device)) {
    printf("No cuda device present.\n");
    return;
  }
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("Major revision number:         %d\n", prop.major);
  printf("Minor revision number:         %d\n", prop.minor);
  printf("Name:                          %s\n", prop.name);
  printf("Total global memory:           %lu\n", prop.totalGlobalMem);
  printf("Total shared memory per block: %lu\n", prop.sharedMemPerBlock);
  printf("Total registers per block:     %d\n", prop.regsPerBlock);
  printf("Warp size:                     %d\n", prop.warpSize);
  printf("Maximum memory pitch:          %lu\n", prop.memPitch);
  printf("Maximum threads per block:     %d\n", prop.maxThreadsPerBlock);
  printf("Maximum dimension of block:    %d, %d, %d\n",
      prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  printf("Maximum dimension of grid:     %d, %d, %d\n",
      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  printf("Clock rate:                    %d\n", prop.clockRate);
  printf("Total constant memory:         %lu\n", prop.totalConstMem);
  printf("Texture alignment:             %lu\n", prop.textureAlignment);
  printf("Concurrent copy and execution: %s\n",
      (prop.deviceOverlap ? "Yes" : "No"));
  printf("Number of multiprocessors:     %d\n", prop.multiProcessorCount);
  printf("Kernel execution timeout:      %s\n",
      (prop.kernelExecTimeoutEnabled ? "Yes" : "No"));
  return;
}

}  // namespace caffe
