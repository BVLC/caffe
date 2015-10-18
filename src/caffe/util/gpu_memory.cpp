#include <vector>
#include "caffe/common.hpp"

#include "caffe/util/gpu_memory.hpp"

// CNMEM integration
#include "cnmem.h"

namespace caffe {

  gpu_memory::PoolMode gpu_memory::mode_ = gpu_memory::NoPool;

#ifdef CPU_ONLY  // CPU-only Caffe.
  void gpu_memory::init(const std::vector<int>& gpus,
                        PoolMode m)  {}
  void gpu_memory::destroy() {}

  const char* gpu_memory::getPoolName()  {
    return "No GPU: CPU Only Memory";
  }
#else
  void gpu_memory::init(const std::vector<int>& gpus,
                           PoolMode m)  {
    if (gpus.size() <= 0) {
      // should we report an error here ?
      m = gpu_memory::NoPool;
    }

    switch (m) {
    case CnMemPool:
      initCNMEM(gpus);
      break;
    case CubPool:
    default:
      break;
    }

    std::cout << "gpu_memory initialized with "
              << getPoolName() << std::endl;
  }

  void gpu_memory::destroy() {
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemFinalize());
      break;
    case CubPool:
    default:
      break;
    }
    mode_ = NoPool;
  }


  void gpu_memory::allocate(void **ptr, size_t size, cudaStream_t stream) {
    CHECK((ptr) != NULL);
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemMalloc(ptr, size, stream));
      break;
    case CubPool:
    default:
      CUDA_CHECK(cudaMalloc(ptr, size));
      break;
    }
  }

  void gpu_memory::deallocate(void *ptr, cudaStream_t stream) {
    // allow for null pointer deallocation
    if (!ptr)
      return;
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemFree(ptr, stream));
      break;
    case CubPool:
    default:
      CUDA_CHECK(cudaFree(ptr));
      break;
    }
  }

  void gpu_memory::registerStream(cudaStream_t stream) {
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemRegisterStream(stream));
      break;
    case CubPool:
    default:
      break;
    }
  }

  void gpu_memory::initCNMEM(const std::vector<int>& gpus) {
#ifdef USE_CNMEM
    cnmemDevice_t* devs = new cnmemDevice_t[gpus.size()];
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
    CUDA_CHECK(cudaSetDevice(initial_device));
    delete [] devs;
    mode_  = CnMemPool;
#endif
  }

  const char* gpu_memory::getPoolName()  {
    switch (mode_) {
    case CnMemPool:
      return "CNMEM Pool";
    case CubPool:
      return "CUB Pool";
    default:
      return "No Pool : Plain CUDA Allocator";
    }
  }

  void gpu_memory::getInfo(size_t *free_mem, size_t *total_mem) {
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemMemGetInfo(free_mem, total_mem, cudaStreamDefault));
      break;
    case CubPool:
    default:
      CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
    }
  }
#endif  // CPU_ONLY

}  // namespace caffe


