#include <algorithm>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/gpu_memory.hpp"


#ifndef CPU_ONLY
#include "cub/cub/util_allocator.cuh"
#endif

namespace caffe {

#ifndef CPU_ONLY  // CPU-only Caffe.
  static cub::CachingDeviceAllocator* cubAlloc = 0;
  vector<gpu_memory::MemInfo> gpu_memory::dev_info_;
#endif

  gpu_memory::PoolMode gpu_memory::mode_   = gpu_memory::NoPool;
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
      initMEM(gpus, m);
      break;
    default:
      break;
    }
    if (debug)
      std::cout << "gpu_memory initialized with "
                << getPoolName()  << std::endl;
  }

  void gpu_memory::destroy() {
    switch (mode_) {
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
    case CubPool:
    default:
      break;
    }
  }

  void gpu_memory::initMEM(const std::vector<int>& gpus, PoolMode m) {
    mode_ = m;
    int initial_device;

    CUDA_CHECK(cudaGetDevice(&initial_device));

    for (int i = 0; i < gpus.size(); i++) {
        int cur_device = gpus[i];
        if (cur_device+1 > dev_info_.size())
            dev_info_.resize(cur_device+1);

      CUDA_CHECK(cudaSetDevice(gpus[i]));
      cudaDeviceProp props;
      CUDA_CHECK(cudaGetDeviceProperties(&props, cur_device));
      CUDA_CHECK(cudaMemGetInfo(&dev_info_[cur_device].free,
                                &dev_info_[cur_device].total));

      if (debug_) {
        std::cout << "cudaGetDeviceProperties: Mem = "
                  << props.totalGlobalMem <<std::endl;
        std::cout << "cudaMemGetInfo_[" << cur_device
                  <<": Free= " << dev_info_[cur_device].free
                  << " Total= " << dev_info_[cur_device].total << std::endl;
      }

      // make sure we don't ask for more that total device memory
      dev_info_[i].free = std::min(dev_info_[cur_device].total,
                                   dev_info_[cur_device].free);
      dev_info_[i].free = std::min(props.totalGlobalMem,
                                   dev_info_[cur_device].free);
    }


    switch ( mode_ ) {
      case CubPool:
        try {
          // just in case someone installed 'no cleanup' arena before
          delete cubAlloc;

          cubAlloc = new cub::CachingDeviceAllocator( 2,
                                                      6,
                                                      16,
                                                      (size_t)-1,
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
  }

  const char* gpu_memory::getPoolName()  {
    switch (mode_) {
    case CubPool:
      return "CUB Pool";
    default:
      return "No Pool : Plain CUDA Allocator";
    }
  }

  void gpu_memory::getInfo(size_t *free_mem, size_t *total_mem) {
    switch (mode_) {
    case CubPool:
      int cur_device;
      CUDA_CHECK(cudaGetDevice(&cur_device));
      *total_mem = dev_info_[cur_device].total;
      // Free memory is initial free memory minus outstanding allocations.
      // Assuming we only allocate via gpu_memory since its constructon.
      *free_mem = dev_info_[cur_device].free -
          cubAlloc->cached_bytes[cur_device].busy;
      break;
    default:
      CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
      break;
    }
  }
#endif  // CPU_ONLY

}  // namespace caffe
