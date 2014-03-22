// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

SyncedMemory::SyncedMemory(const size_t size) : cpu_vector_(0),
    gpu_vector_(0), size_(size), head_(UNINITIALIZED) {
}

SyncedMemory::~SyncedMemory() {
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    cpu_vector_.resize(size_, 0);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
    if (cpu_vector_.size() < size_) {
      cpu_vector_.resize(size_, 0);
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
    gpu_vector_.resize(size_, 0);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_vector_.size() < size_) {
      gpu_vector_.resize(size_, 0);
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

