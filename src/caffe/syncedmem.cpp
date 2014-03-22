// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
}

SyncedMemory::~SyncedMemory() {
}

/*
 * http://thrust.github.io/doc/classthrust_1_1host__vector.html
 * http://thrust.github.io/doc/classthrust_1_1device__vector.html
 * thrust::host_vector and thrust::device_vector will resize this vector to 
 *   the specified number of elements. If the number is smaller than this 
 *   vector's current size this vector is truncated, otherwise this vector 
 *   is extended and new elements are populated with given data.
 */
void SyncedMemory::resize(const size_t size, const uint8_t default_value) {
  size_ = size;
  resize_default_value_ = default_value;
  reserve(size);
}

/*
 * http://thrust.github.io/doc/classthrust_1_1host__vector.html
 * http://thrust.github.io/doc/classthrust_1_1device__vector.html
 * If n is less than or equal to capacity(), this call has no effect. 
 * Otherwise, this method is a request for allocation of additional memory. 
 * If the request is successful, then capacity() is greater than or equal to n;
 *   otherwise, capacity() is unchanged. In either case, size() is unchanged.
 */
void SyncedMemory::reserve(const size_t capacity) {
  if (capacity_ < capacity) {
    capacity_ = capacity;
  }
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    cpu_vector_.resize(size_, resize_default_value_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    cpu_vector_ = gpu_vector_;
    head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  switch (head_) {
  case UNINITIALIZED:
    gpu_vector_.resize(size_, resize_default_value_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (cpu_vector_.size() < size_) {
      cpu_vector_.resize(size_, resize_default_value_);
    }
    if (gpu_vector_.capacity() < cpu_vector_.size()) {
      gpu_vector_.reserve(cpu_vector_.size());
    }
    gpu_vector_ = cpu_vector_;
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)thrust::raw_pointer_cast(cpu_vector_.data());
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
  to_gpu();
  return (const void*)thrust::raw_pointer_cast(gpu_vector_.data());
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return (void*)thrust::raw_pointer_cast(cpu_vector_.data());
}

void* SyncedMemory::mutable_gpu_data() {
  to_gpu();
  head_ = HEAD_AT_GPU;
  return (void*)thrust::raw_pointer_cast(gpu_vector_.data());
}


}  // namespace caffe

