#include <algorithm>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/gpu_memory.hpp"

#ifndef CPU_ONLY
#include "cub/util_allocator.cuh"
#endif

namespace caffe {

#ifndef CPU_ONLY
static cub::CachingDeviceAllocator* cub_allocator = 0;
vector<GPUMemoryManager::MemInfo> GPUMemoryManager::dev_info_;
#endif

GPUMemoryManager::PoolMode GPUMemoryManager::mode_ = GPUMemoryManager::NO_POOL;
bool GPUMemoryManager::debug_ = false;

#ifdef CPU_ONLY
void GPUMemoryManager::init(const std::vector<int>&, PoolMode, bool) {}
void GPUMemoryManager::destroy() {}

const char* GPUMemoryManager::pool_name() {
  return "No GPU: CPU Only Memory";
}
#else

void GPUMemoryManager::init(const std::vector<int>& gpus, PoolMode m,
    bool debug) {
  bool debug_env = getenv("DEBUG_GPU_MEM") != 0;
  debug_ = debug || debug_env;
  if (gpus.size() <= 0) {
    m = GPUMemoryManager::NO_POOL;
  }

  switch (m) {
  case CUB_POOL:
    InitMemory(gpus, m);
    break;
  default:
    break;
  }
  VLOG_IF(1, debug) << "GPUMemoryManager initialized with " << pool_name();
}

void GPUMemoryManager::destroy() {
  switch (mode_) {
  case CUB_POOL:
    delete cub_allocator;
    cub_allocator = NULL;
    break;
  default:
    break;
  }
  mode_ = NO_POOL;
}

bool GPUMemoryManager::try_allocate(void** ptr, size_t size,
    cudaStream_t stream) {
  CHECK((ptr) != NULL);
  cudaError_t status = cudaSuccess, last_err = cudaSuccess;
  switch (mode_) {
  case CUB_POOL:
    // Clean Cache & Retry logic is inside now
    status = cub_allocator->DeviceAllocate(ptr, size, stream);
    // If there was a retry and it succeeded we get good status here but
    // we need to clean up last error...
    last_err = cudaGetLastError();
    // ...and update the dev info if something was wrong
    if (status != cudaSuccess || last_err != cudaSuccess) {
      int cur_device;
      CUDA_CHECK(cudaGetDevice(&cur_device));
      // Refresh per-device saved values.
      for (int i = 0; i < dev_info_.size(); ++i) {
        // only query devices that were initialized
        if (dev_info_[i].total_) {
          update_dev_info(i);
          // record which device caused cache flush
          if (i == cur_device) {
            dev_info_[i].flush_count_++;
          }
        }
      }
    }
    break;
  default:
    status = cudaMalloc(ptr, size);
    break;
  }
  return status == cudaSuccess;
}

void GPUMemoryManager::deallocate(void* ptr, cudaStream_t stream) {
  // allow for null pointer deallocation
  if (!ptr) {
    return;
  }
  switch (mode_) {
  case CUB_POOL:
    CUDA_CHECK(cub_allocator->DeviceFree(ptr));
    break;
  default:
    CUDA_CHECK(cudaFree(ptr));
    break;
  }
}

void GPUMemoryManager::update_dev_info(int device) {
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  if (device + 1 > dev_info_.size()) {
    dev_info_.resize(device + 1);
  }

  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  CUDA_CHECK(cudaMemGetInfo(&dev_info_[device].free_,
      &dev_info_[device].total_));

  VLOG_IF(1, debug_) << "cudaGetDeviceProperties: Mem = "
                     << props.totalGlobalMem;
  VLOG_IF(1, debug_) << "cudaMemGetInfo_[" << device << "]: Free="
                     << dev_info_[device].free_ << " Total="
                     << dev_info_[device].total_;

  // Make sure we don't have more that total device memory
  dev_info_[device].total_ = std::min(props.totalGlobalMem,
      dev_info_[device].total_);
  // Here we are adding existing 'busy' allocations to CUDA free memory
  dev_info_[device].free_ = std::min(dev_info_[device].total_,
      dev_info_[device].free_ + cub_allocator->cached_bytes[device].live);
  CUDA_CHECK(cudaSetDevice(initial_device));
}

void GPUMemoryManager::InitMemory(const std::vector<int>& gpus, PoolMode m) {
  mode_ = m;
  switch (mode_) {
  case CUB_POOL:
    try {
      // Just in case someone installed 'no cleanup' arena before
      delete cub_allocator;
      cub_allocator = new cub::CachingDeviceAllocator(2, 6, 22, (size_t) -1,
          false, debug_);
    } catch (...) {
    }
    CHECK(cub_allocator);
    for (int i = 0; i < gpus.size(); ++i) {
      update_dev_info(gpus[i]);
    }
    break;
  default:
    break;
  }
}

const char* GPUMemoryManager::pool_name() {
  switch (mode_) {
  case CUB_POOL:
    return "CUB Pool";
  default:
    return "No Pool: Plain CUDA Allocator";
  }
}

void GPUMemoryManager::GetInfo(size_t* free_mem, size_t* total_mem) {
  switch (mode_) {
  case CUB_POOL:
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    *total_mem = dev_info_[cur_device].total_;
    // Free memory is initial free memory minus outstanding allocations.
    // Assuming we only allocate via GPUMemoryManager since its construction.
    *free_mem = dev_info_[cur_device].free_ -
        cub_allocator->cached_bytes[cur_device].live;
    break;
  default:
    CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
    break;
  }
}
#endif  // CPU_ONLY

}  // namespace caffe
