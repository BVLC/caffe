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
    bool debug_env = (getenv("DEBUG_GPU_MEM") != 0);
    debug_ = debug || debug_env;

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
    VLOG_IF(1, debug) << "gpu_memory initialized with " << getPoolName();
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
      if (cubAlloc->DeviceAllocate(ptr, size, stream) != cudaSuccess) {
          int cur_device;
          CUDA_CHECK(cudaGetDevice(&cur_device));
          // free all cached memory (for all devices), synchrionize
          cudaDeviceSynchronize();
          cudaThreadSynchronize();
          cubAlloc->FreeAllCached();
          cudaDeviceSynchronize();
          cudaThreadSynchronize();

          // Refresh per-device saved values.
          for (int i = 0; i < dev_info_.size(); i++) {
            // only query devices that were initialized
            if (dev_info_[i].total) {
              update_dev_info(i);
              // record which device caused cache flush
              if (i == cur_device)
                dev_info_[i].flush_count++;
            }
          }
          // retry once
          CUDA_CHECK(cubAlloc->DeviceAllocate(ptr, size, stream));
          // If retry succeeds we need to clean up last error:
          cudaGetLastError();
        }
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


  void gpu_memory::update_dev_info(int device) {
    int initial_device;
    CUDA_CHECK(cudaGetDevice(&initial_device));

    if (device+1 > dev_info_.size())
      dev_info_.resize(device+1);

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    CUDA_CHECK(cudaMemGetInfo(&dev_info_[device].free,
                              &dev_info_[device].total));

    VLOG_IF(1, debug_) << "cudaGetDeviceProperties: Mem = "
                       << props.totalGlobalMem;
    VLOG_IF(1, debug_) << "cudaMemGetInfo_[" << device
                       << "]: Free=" << dev_info_[device].free
                       << " Total=" << dev_info_[device].total;

    // make sure we don't have more that total device memory
    dev_info_[device].total = std::min(props.totalGlobalMem,
                                           dev_info_[device].total);

    // here we are adding existing 'busy' allocations to CUDA free memory
    dev_info_[device].free =
      std::min(dev_info_[device].total,
               dev_info_[device].free
               + cubAlloc->cached_bytes[device].busy);
    CUDA_CHECK(cudaSetDevice(initial_device));
  }

  void gpu_memory::initMEM(const std::vector<int>& gpus, PoolMode m) {
    mode_ = m;
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
        for (int i = 0; i < gpus.size(); i++) {
          update_dev_info(gpus[i]);
        }
        break;
      default:
        break;
      }
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
