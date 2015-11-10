#include <algorithm>
#include <vector>
#include "caffe/common.hpp"

#include "caffe/util/gpu_memory.hpp"


#ifdef USE_CNMEM
// CNMEM integration
#include "cnmem.h"
#endif

#ifndef CPU_ONLY
#include "cub/cub/util_allocator.cuh"
#endif

namespace caffe {

#ifndef CPU_ONLY  // CPU-only Caffe.
  static cub::CachingDeviceAllocator* cubAlloc = 0;
#endif

  gpu_memory::PoolMode gpu_memory::mode_   = gpu_memory::NoPool;
  size_t               gpu_memory::poolsize_ = 0;
  bool                 gpu_memory::debug_ = false;

#ifdef CPU_ONLY  // CPU-only Caffe.
  void gpu_memory::init(const std::vector<int>&, PoolMode, bool) {}
  void gpu_memory::destroy() {}

  const char* gpu_memory::getPoolName()  {
    return "No GPU: CPU Only Memory";
  }
#else
  void gpu_memory::init(const std::vector<int>& gpus,
                        PoolMode m, bool debug) {
    debug_ = debug;
    if (gpus.size() <= 0) {
      // should we report an error here ?
      m = gpu_memory::NoPool;
    }

    switch (m) {
    case CubPool:
    case CnMemPool:
      initMEM(gpus, m);
      break;
    default:
      break;
    }
    if (debug)
      std::cout << "gpu_memory initialized with "
                << getPoolName() << ". Poolsize = "
                << (1.0*poolsize_)/(1024.0*1024.0*1024.0) << " G." << std::endl;
  }

  void gpu_memory::destroy() {
    switch (mode_) {
    case CnMemPool:
      CNMEM_CHECK(cnmemFinalize());
      break;
    case CubPool:
      delete cubAlloc;
      cubAlloc = NULL;
      break;
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
      CUDA_CHECK(cubAlloc->DeviceAllocate(ptr, size, stream));
      break;
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
      CUDA_CHECK(cubAlloc->DeviceFree(ptr));
      break;
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

  void gpu_memory::initMEM(const std::vector<int>& gpus, PoolMode m) {
    mode_ = m;
    int initial_device;
#if USE_CNMEM
    cnmemDevice_t* devs = new cnmemDevice_t[gpus.size()];
#endif

    CUDA_CHECK(cudaGetDevice(&initial_device));

    for (int i = 0; i < gpus.size(); i++) {
      CUDA_CHECK(cudaSetDevice(gpus[i]));
      size_t free_mem, total_mem;
      cudaDeviceProp props;
      CUDA_CHECK(cudaGetDeviceProperties(&props, gpus[i]));
      CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

      if (debug_) {
        std::cout << "cudaGetDeviceProperties: Mem = "
                  << props.totalGlobalMem <<std:: endl;
        std::cout << "cudaMemGetInfo: Free= " << free_mem
                  << " Total= " << total_mem << std::endl;
      }

      // make sure we don't ask for more that total device memory
      free_mem = std::min(total_mem, free_mem);
      free_mem = size_t(0.95*std::min(props.totalGlobalMem, free_mem));
      // find out the smallest GPU size
      if (poolsize_ == 0 || poolsize_ > free_mem)
        poolsize_ = free_mem;
#if USE_CNMEM
      devs[i].device = gpus[i];
      devs[i].size = free_mem;
      devs[i].numStreams = 0;
      devs[i].streams = NULL;
#endif
    }


    switch ( mode_ ) {
      case CnMemPool:
#if USE_CNMEM
        CNMEM_CHECK(cnmemInit(gpus.size(), devs, CNMEM_FLAGS_DEFAULT));
#endif
        break;
      case CubPool:
        try {
          // if you are paranoid, that doesn't mean they are not after you :)
          delete cubAlloc;

          cubAlloc = new cub::CachingDeviceAllocator( 2,   // defaults
                                                      6,
                                                      16,
                                                      poolsize_,
                                                      false,
                                                      debug_);
        }
        catch (...) {}
        CHECK(cubAlloc);
        break;
      default:
        break;
      }

    CUDA_CHECK(cudaSetDevice(initial_device));
#if USE_CNMEM
    delete [] devs;
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
      int cur_device;
      CUDA_CHECK(cudaGetDevice(&cur_device));
      *total_mem = poolsize_;
      // Free memory is initial free memory minus outstanding allocations.
      // Assuming we only allocate via gpu_memory since its constructon.
      *free_mem = poolsize_ - cubAlloc->cached_bytes[cur_device].busy;
      break;
    default:
      CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
    }
  }
#endif  // CPU_ONLY

}  // namespace caffe
