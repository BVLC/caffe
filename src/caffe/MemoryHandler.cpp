#include "caffe/common.hpp"
#include "caffe/MemoryHandler.hpp"

#include <boost/thread.hpp>

namespace caffe {

bool MemoryHandler::using_pool_ = false;
bool MemoryHandler::initialized_ = false;
std::vector<int> MemoryHandler::gpus_;

using namespace boost;

#ifndef CPU_ONLY  // CPU-only Caffe.

static boost::mutex memHandlerMutex;
static boost::shared_ptr<MemoryHandler> mem_handler;

MemoryHandler& MemoryHandler::Get() {
  if (!mem_handler.get()) {
    boost::mutex::scoped_lock lock(memHandlerMutex);
    if (!mem_handler.get()) {
      mem_handler.reset(new MemoryHandler());
    }
  }
  return *(mem_handler.get());
}

void MemoryHandler::mallocGPU(void **ptr, size_t size, cudaStream_t stream) {
  Get().allocate_memory(ptr, size, stream);
}


void MemoryHandler::freeGPU(void *ptr, cudaStream_t stream) {
  Get().free_memory(ptr, stream);
}

void MemoryHandler::allocate_memory(void **ptr, size_t size,
                                    cudaStream_t stream) {
  CHECK(initialized_);
  int initial_device;
  cudaGetDevice(&initial_device);
  if (using_pool_) {
#ifdef USE_CNMEM
    CNMEM_CHECK(cnmemMalloc(ptr, size, stream));
#endif
  } else {
    CUDA_CHECK(cudaMalloc(ptr, size));
  }
  cudaSetDevice(initial_device);
}

void MemoryHandler::free_memory(void *ptr, cudaStream_t stream) {
  CHECK(initialized_);
  // boost::mutex::scoped_lock lock(memHandlerMutex);
  int initial_device;
  cudaGetDevice(&initial_device);
  if (using_pool_) {
#ifdef USE_CNMEM
    CNMEM_CHECK(cnmemFree(ptr, stream));
#endif
  } else {
    CUDA_CHECK(cudaFree(ptr));
  }
  cudaSetDevice(initial_device);
}

void MemoryHandler::registerStream(cudaStream_t stream) {
  if (Get().using_pool_) {
#ifdef USE_CNMEM
    CNMEM_CHECK(cnmemRegisterStream(stream));
#endif
  }
}

void MemoryHandler::destroy() {
  CHECK(initialized_);
#ifdef USE_CNMEM
  CNMEM_CHECK(cnmemFinalize());
#endif
}

void MemoryHandler::Init() {
  CHECK(!initialized_);
#ifdef USE_CNMEM
  if (Get().using_pool_) {
    cnmemDevice_t *devs = new cnmemDevice_t[Get().gpus_.size()];

    int initial_device;
    CUDA_CHECK(cudaGetDevice(&initial_device));

    for (int i = 0; i < Get().gpus_.size(); i++) {
      CUDA_CHECK(cudaSetDevice(Get().gpus_[i]));

      devs[i].device = Get().gpus_[i];

      size_t free_mem, used_mem;
      CUDA_CHECK(cudaMemGetInfo(&free_mem, &used_mem));

      devs[i].size = size_t(0.95*free_mem);
      devs[i].numStreams = 0;
      devs[i].streams = NULL;
    }
    CNMEM_CHECK(cnmemInit(Get().gpus_.size(), devs, CNMEM_FLAGS_DEFAULT));
    Get().initialized_ = true;

    CUDA_CHECK(cudaSetDevice(initial_device));

    delete [] devs;
  }
#endif
  Get().initialized_ = true;
}

void MemoryHandler::getInfo(size_t *free_mem, size_t *total_mem) {
  if (Get().using_pool_) {
#ifdef USE_CNMEM
    CNMEM_CHECK(cnmemMemGetInfo(free_mem, total_mem, cudaStreamDefault));
#endif
  } else {
    CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
  }
}

const char* cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

}

#endif  // CPU_ONLY

