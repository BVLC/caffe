#include "caffe/common.hpp"
#include "caffe/MemoryHandler.hpp"

#include <boost/thread.hpp>

namespace caffe {

bool MemoryHandler::using_pool_ = false;
bool MemoryHandler::initialized_ = false;

using namespace boost;

#ifndef CNMEM_CHECK
#  define CNMEM_CHECK(x) 
#endif

#ifndef CPU_ONLY  // CPU-only Caffe.

void MemoryHandler::mallocGPU(void **ptr, size_t size, cudaStream_t stream) {
  CHECK(initialized_);
  if (using_pool_) {
    CNMEM_CHECK(cnmemMalloc(ptr, size, stream));
  } else {
    CUDA_CHECK(cudaMalloc(ptr, size));
  }
}


void MemoryHandler::freeGPU(void *ptr, cudaStream_t stream) {
  CHECK(initialized_);
  if (using_pool_) {
    CNMEM_CHECK(cnmemFree(ptr, stream));
  } else {
    CUDA_CHECK(cudaFree(ptr));
  }
}

void MemoryHandler::registerStream(cudaStream_t stream) {
  CHECK(initialized_);
  if (using_pool_) {
    CNMEM_CHECK(cnmemRegisterStream(stream));
  }
}

void MemoryHandler::destroy() {
  CHECK(initialized_);
  CNMEM_CHECK(cnmemFinalize());
  initialized_ = false;
  using_pool_ = false;
}

void MemoryHandler::init(const std::vector<int>& gpus, bool use_pool) {
  CHECK(!initialized_);
#ifdef USE_CNMEM
  if (false /* use_pool */) {
     using_pool_  = true;
    cnmemDevice_t *devs = new cnmemDevice_t[gpus.size()];

    int initial_device;
    CUDA_CHECK(cudaGetDevice(&initial_device));

    for (int i = 0; i < gpus.size(); i++) {
      CUDA_CHECK(cudaSetDevice(gpus[i]));

      devs[i].device = gpus[i];

      size_t free_mem, used_mem;
      CUDA_CHECK(cudaMemGetInfo(&free_mem, &used_mem));

      devs[i].size = size_t(0.95*free_mem);
      devs[i].numStreams = 0;
      devs[i].streams = NULL;
    }
    CNMEM_CHECK(cnmemInit(gpus.size(), devs, CNMEM_FLAGS_DEFAULT));
    initialized_ = true;

    CUDA_CHECK(cudaSetDevice(initial_device));

    delete [] devs;
  }
#endif
  initialized_ = true;
  std::cout << "MemoryHandler initialized" << 
    (using_pool_ ? " with CNMEM pool.\n" : " with CUDA allocator.\n");
}

void MemoryHandler::getInfo(size_t *free_mem, size_t *total_mem) {
  if (using_pool_) {
    CNMEM_CHECK(cnmemMemGetInfo(free_mem, total_mem, cudaStreamDefault));
  } else {
    CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
  }
}

}

#endif  // CPU_ONLY

