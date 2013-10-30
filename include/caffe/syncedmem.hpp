// Copyright 2013 Yangqing Jia

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {


#if 0

// This chunk of code should be used when one has a machine that does not have
// GPU, thus cannot be used if we just want to distribute a single binary.

inline void CaffeMallocHost(void** ptr, size_t size) {
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
  } else {
    *ptr = malloc(size);
  }
}

inline void CaffeFreeHost(void* ptr) {
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaFreeHost(ptr));
  } else {
    free(ptr);
  }
}

#else

// This chunk of code is safer, but may not be as fast as the cuda pinned memory
// version.

inline void CaffeMallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
}

inline void CaffeFreeHost(void* ptr) {
  free(ptr);
}

#endif  // code to define CaffeMallocHost and CaffeFreeHost

class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED) {}
  ~SyncedMemory();
  const void* cpu_data();
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }
 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
