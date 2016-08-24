#ifndef CPU_ONLY
#include <algorithm>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/util/gpu_memory.hpp"

#include "cub/util_allocator.cuh"

namespace caffe {
using std::vector;

const int GPUMemory::INVALID_DEVICE =
    cub::CachingDeviceAllocator::INVALID_DEVICE_ORDINAL;
const unsigned int GPUMemory::Manager::BIN_GROWTH = 2;
const unsigned int GPUMemory::Manager::MIN_BIN = 6;
const unsigned int GPUMemory::Manager::MAX_BIN = 22;
const size_t GPUMemory::Manager::MAX_CACHED_BYTES = (size_t) -1;

GPUMemory::Manager GPUMemory::mgr_;

GPUMemory::Manager::Manager()
  : mode_(CUDA_MALLOC), debug_(false), initialized_(false),
    cub_allocator_(NULL) {}

void GPUMemory::Manager::init(const vector<int>& gpus, Mode m, bool debug) {
  if (initialized_) {
    return;
  }
  bool debug_env = getenv("DEBUG_GPU_MEM") != 0;
  debug_ = debug || debug_env;
  if (gpus.size() <= 0) {
    m = CUDA_MALLOC;
  }
  switch (m) {
  case CUB_ALLOCATOR:
    try {
      // Just in case someone installed 'no cleanup' arena before
      delete cub_allocator_;
      cub_allocator_ = new cub::CachingDeviceAllocator(BIN_GROWTH, MIN_BIN,
          MAX_BIN, MAX_CACHED_BYTES, true, debug_);
    } catch (...) {
    }
    CHECK_NOTNULL(cub_allocator_);
    for (int i = 0; i < gpus.size(); ++i) {
      update_dev_info(gpus[i]);
    }
    break;
  default:
    break;
  }
  mode_ = m;
  initialized_ = true;
  DLOG(INFO) << "GPUMemory::Manager initialized with " << pool_name();
}

GPUMemory::Manager::~Manager() {
  switch (mode_) {
  case CUB_ALLOCATOR:
    {
      vector<cudaStream_t>::iterator it = device_streams_.begin();
      while (it != device_streams_.end()) {
        cudaStreamDestroy(*it++);
      }
    }
    delete cub_allocator_;
    break;
  default:
    break;
  }
}

bool GPUMemory::Manager::try_allocate(void** ptr, size_t size, int device,
    cudaStream_t stream) {
  CHECK(initialized_) << "Create GPUMemory::Scope to initialize Memory Manager";
  CHECK_NOTNULL(ptr);
  cudaError_t status = cudaSuccess, last_err = cudaSuccess;
  switch (mode_) {
  case CUB_ALLOCATOR:
    // Clean Cache & Retry logic is inside now
    status = cub_allocator_->DeviceAllocate(device, ptr, size, stream);
    // If there was a retry and it succeeded we get good status here but
    // we need to clean up last error...
    last_err = cudaGetLastError();
    // ...and update the dev info if something was wrong
    if (status != cudaSuccess || last_err != cudaSuccess) {
      // If we know what particular device failed we update its info only
      if (device > INVALID_DEVICE && device < dev_info_.size()) {
        // only query devices that were initialized
        if (dev_info_[device].total_) {
          update_dev_info(device);
          dev_info_[device].flush_count_++;
        }
      } else {
        // Update them all otherwise
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
    }
    break;
  default:
    status = cudaMalloc(ptr, size);
    break;
  }
  return status == cudaSuccess;
}

void GPUMemory::Manager::deallocate(void* ptr, int device,
    cudaStream_t stream) {
  // allow for null pointer deallocation
  if (!ptr) {
    return;
  }
  switch (mode_) {
  case CUB_ALLOCATOR:
    {
      int current_device;
      cudaError_t status = cudaGetDevice(&current_device);
      // Preventing dead lock while Caffe shutting down.
      if (status != cudaErrorCudartUnloading) {
        CUDA_CHECK(cub_allocator_->DeviceFree(device, ptr));
      }
    }
    break;
  default:
    CUDA_CHECK(cudaFree(ptr));
    break;
  }
}

void GPUMemory::Manager::update_dev_info(int device) {
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

  DLOG(INFO) << "cudaGetDeviceProperties: Mem = " << props.totalGlobalMem;
  DLOG(INFO) << "cudaMemGetInfo_[" << device << "]: Free="
             << dev_info_[device].free_ << " Total="
             << dev_info_[device].total_;

  // Make sure we don't have more than total device memory.
  dev_info_[device].total_ = std::min(props.totalGlobalMem,
      dev_info_[device].total_);
  dev_info_[device].free_ = std::min(dev_info_[device].total_,
      dev_info_[device].free_);
  CUDA_CHECK(cudaSetDevice(initial_device));
}

const char* GPUMemory::Manager::pool_name() const {
  switch (mode_) {
  case CUB_ALLOCATOR:
    return "Caching (CUB) GPU Allocator";
  default:
    return "Plain CUDA GPU Allocator";
  }
}

void GPUMemory::Manager::GetInfo(size_t* free_mem, size_t* total_mem) {
  switch (mode_) {
  case CUB_ALLOCATOR:
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    *total_mem = dev_info_[cur_device].total_;
    // Free memory is free GPU memory plus free cached memory in the pool.
    *free_mem = dev_info_[cur_device].free_ +
        cub_allocator_->cached_bytes[cur_device].free;
    if (*free_mem > *total_mem) {  // sanity check
      *free_mem = *total_mem;
    }
    break;
  default:
    CUDA_CHECK(cudaMemGetInfo(free_mem, total_mem));
    break;
  }
}

shared_ptr<GPUMemory::Workspace>
GPUMemory::MultiWorkspace::current_workspace(int device) const {
  if (device + 1 > workspaces_.size()) {
    workspaces_.resize(device + 1);
  }
  shared_ptr<GPUMemory::Workspace>& ws = workspaces_[device];
  if (!ws) {  // In case if --gpu=1,0
    ws.reset(new GPUMemory::Workspace);
  }
  return ws;
}

cudaStream_t
GPUMemory::Manager::device_stream(int device) {
  if (device + 1 > device_streams_.size()) {
    device_streams_.resize(device + 1);
  }
  cudaStream_t& stream = device_streams_[device];
  if (!stream) {
    // Here we assume that device is current.
    CUDA_CHECK(cudaStreamCreate(&stream));
  }
  return stream;
}

}  // namespace caffe

#endif  // CPU_ONLY
