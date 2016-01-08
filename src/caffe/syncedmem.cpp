#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

  if (prv_ptr_  && own_prv_data_) {
    CaffeFreeHost(prv_ptr_);
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
#pragma omp critical
    {
      if (head_ == UNINITIALIZED) {
        CaffeMallocHost(&cpu_ptr_, size_);
        caffe_memset(size_, 0, cpu_ptr_);
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = true;
      }
    }
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
  case HEAD_AT_PRV:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    if(NULL == sync_prv_to_cpu_)
    {
      LOG(FATAL) << " Can't sync prv data to cpu";
      //memcpy(cpu_ptr_, prv_ptr_, size_);
    }
    else
      sync_prv_to_cpu_(prv_ptr_, cpu_ptr_, prv_descriptor_);
    head_ = SYNCED_PRV;
    break;
  case SYNCED_PRV:
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
  case HEAD_AT_PRV:
    to_cpu();
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
#pragma omp critical
  {
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
  }
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

/*
  If data is NULL, then allocate the memory here.
  same_data - shall be true if data will be the same as in cpu_ptr_
    but (potentially) with different layout.
*/
void SyncedMemory::set_prv_data(void* data, bool same_data) {
#pragma omp critical
  {
    if(data != NULL) {
      if (prv_ptr_ && own_prv_data_) {
        CaffeFreeHost(prv_ptr_);
      }
      prv_ptr_ = data;
    }
    else if(NULL == prv_ptr_) {
      CaffeMallocHost(&prv_ptr_, size_);
      caffe_memset(size_, 0, prv_ptr_);
    }

    if(same_data)
      head_ = SYNCED_PRV;
    else
      head_ = HEAD_AT_PRV;

    own_prv_data_ = false;
  }
}

const void* SyncedMemory::prv_data() {

  if((head_ != HEAD_AT_PRV) &&
     (head_ != SYNCED_PRV)) {
    // Call set_prv_data() or init_prv_data() first
    return NULL;
  }

  return (const void* )prv_ptr_;
}

void* SyncedMemory::mutable_prv_data() {
  head_ = HEAD_AT_PRV;

  if(NULL == prv_ptr_) {
    CaffeMallocHost(&prv_ptr_, size_);
    caffe_memset(size_, 0, prv_ptr_);
  }
  return prv_ptr_;
}

}  // namespace caffe

