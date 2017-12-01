#include <map>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#ifndef CPU_ONLY
#include <deepir/allocator/buddy_pool.hpp>
#endif

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#ifndef CPU_ONLY

#endif

namespace caffe {

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
    host_free(cpu_ptr_, size_);
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
    host_malloc(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      host_malloc(&cpu_ptr_, size_);
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
    // TODO 去除memset
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
    host_free(cpu_ptr_, size_);
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

#ifndef CPU_ONLY
void *SyncedMemory::gpu_malloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  void *ptr = nullptr;

  const auto &pool = Caffe::device_pool();
  if (pool.get()) {
    ptr = pool->alloc(size);
  } else {
    printf("no memory pool\n");
  }

  if (ptr) {
    device_pool_ = pool;
  } else {
    printf("alloc from cuda size=%zu\n", size);
    CUDA_CHECK(cudaMalloc(&ptr, size));
  }

  //  std::cout<<std::this_thread::get_id()<<" malloc "<<(size_t)ptr<<std::endl;
  return ptr;
}

void SyncedMemory::gpu_free(void *data) {
  if (!data) {
    return;
  }

  if (device_pool_.get()) {
    CHECK(device_pool_->free(data)) << "free device failed";
    device_pool_.reset();
    return;
  }
  CUDA_CHECK(cudaFree(data));

  /*
  CUDA_CHECK(cudaDeviceSynchronize());
  //在caffe實例銷毀的時候由其它線程觸發刪除
   puts("gpu_free global");
  auto device_id = Caffe::GetDevice();

//  std::shared_lock<std::shared_timed_mutex> lock(gpu_pool_mutex);
  for(auto &pool:device_pools[device_id]) {
    if(pool->free(data)) {
      return;
    }
  }
  */
}

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
void SyncedMemory::host_malloc(void **ptr, size_t size) {
#ifndef CPU_ONLY
  constexpr size_t pinned_memory_max_size = 128;
  if (Caffe::mode() == Caffe::GPU && size <= pinned_memory_max_size) {
    auto const &pool = Caffe::host_pool();
    if (pool.get()) {
      *ptr = pool->alloc(size);
      if (*ptr) {
        // puts("malloc cubud cpu");
        host_pool_ = pool;
        cpu_malloc_use_cuda_ = true;
        return;
      }
    }
    if (cudaMallocHost(ptr, size) == cudaSuccess) {
      cpu_malloc_use_cuda_ = true;
      return;
    }
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size : 1, 64);
#else
  *ptr = malloc(size);
#endif
  cpu_malloc_use_cuda_ = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

void SyncedMemory::host_free(void *ptr, size_t size) {
#ifndef CPU_ONLY
  if (cpu_malloc_use_cuda_) {
    if (host_pool_.get()) {
      CHECK(host_pool_->free(ptr)) << "free host failed";
      host_pool_.reset();
      return;
    }
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

#endif

} // namespace caffe
