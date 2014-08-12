#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
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
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
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

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
#endif
}


}  // namespace caffe

