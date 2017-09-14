#include <deepir/cuda_buddy_pool.hpp>
#include <memory>
#include <shared_mutex>
#include <stdexcept>

#include "caffe/common.hpp"
#include "caffe/gpu_memory_pool.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

static constexpr size_t device_max_num = 32;
static constexpr size_t pinned_memory_max_size = 128;

static std::array<std::unique_ptr<deepir::cuda_buddy_pool>, device_max_num>
    device_gpu_pools;

static std::shared_timed_mutex gpu_pool_mutex;

namespace caffe {

void set_gpu_memory_pool(size_t memory_bytes) {
  uint8_t max_level = 27;
  size_t num = memory_bytes / ((size_t)1 << max_level);
  if (num == 0) {
    num++;
  }

  auto device_id = Caffe::GetDevice();
  CHECK(device_id >= 0 && device_id < device_max_num);
  std::lock_guard<std::shared_timed_mutex> lock(gpu_pool_mutex);
  if (!device_gpu_pools[device_id]) {
    device_gpu_pools[device_id] = std::make_unique<deepir::cuda_buddy_pool>(
        num, max_level, deepir::cuda_buddy_pool::alloc_location::device);
  } else {
    throw std::runtime_error(std::string("caffe has gpu allocator on device ") +
                             std::to_string(device_id));
  }
}

static inline deepir::cuda_buddy_pool *get_gpu_memory_pool() {
  auto device_id = Caffe::GetDevice();
  CHECK(device_id >= 0 && device_id < device_max_num);
  std::shared_lock<std::shared_timed_mutex> lock(gpu_pool_mutex);
  return device_gpu_pools[device_id] ? device_gpu_pools[device_id].get()
                                     : nullptr;
}

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
static inline void CaffeMallocHost(void **ptr, size_t size, bool *use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU && size <= pinned_memory_max_size) {
    if (cudaMallocHost(ptr, size) == cudaSuccess) {
      *use_cuda = true;
      return;
    }
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size : 1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

static inline void CaffeFreeHost(void *ptr, size_t size, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}

size_t SyncedMemory::get_used_size() { return 0; }
SyncedMemory::SyncedMemory()
    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
    : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
      own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, size_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    gpu_free(gpu_ptr_);
  }
#endif // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    gpu_ptr_ = gpu_malloc(size_);
    CUDA_CHECK(cudaMemsetAsync(gpu_ptr_, 0, size_, cudaStreamPerThread));
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      gpu_ptr_ = gpu_malloc(size_);
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void *SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void *)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void *data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, size_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void *SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void *)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void *data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    gpu_free(gpu_ptr_);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void *SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void *SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

void *SyncedMemory::gpu_malloc(size_t size) {
  void *ptr = nullptr;

  auto pool = get_gpu_memory_pool();
  if (pool) {
    ptr = pool->alloc_with_lock(size);
  }

  if (!ptr) {
    CUDA_CHECK(cudaMalloc(&ptr, size));
  }
  return ptr;
}

void SyncedMemory::gpu_free(void *data) {
  if (!data) {
    return;
  }
  auto pool = get_gpu_memory_pool();
  if (!pool || !pool->free_with_lock(data)) {
    CUDA_CHECK(cudaFree(data));
  }
}

} // namespace caffe
