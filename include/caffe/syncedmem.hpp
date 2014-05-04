// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

// Theoretically, CaffeMallocHost and CaffeFreeHost should simply call the
// cudaMallocHost and cudaFree functions in order to create pinned memory.
// However, those codes rely on the existence of a cuda GPU (I don't know
// why that is a must since allocating memory should not be accessing the
// GPU resorce, but it just creates an error as of Cuda 5.0) and will cause
// problem when running on a machine without GPU. Thus, we simply define
// these two functions for safety and possible future change if the problem
// of calling cuda functions disappears in a future version.
//
// In practice, although we are creating unpinned memory here, as long as we
// are constantly accessing them the memory pages almost always stays in
// the physical memory (assuming we have large enough memory installed), and
// does not seem to create a memory bottleneck here.

inline void CaffeMallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
  CHECK(*ptr) << "malloc failed when attempting to allocate "
              << size << " bytes of host memory.";
}

inline void CaffeReallocHost(void** ptr, size_t size) {
  *ptr = realloc(*ptr, size);
  CHECK(*ptr) << "realloc failed when attempting to allocate "
              << size << " bytes of host memory.";
}

inline void CaffeFreeHost(void* ptr) {
  free(ptr);
}


class SyncedMemory {
 public:
  explicit SyncedMemory(const size_t size = 0)
      : cpu_data_(NULL), gpu_data_(NULL), size_(size),
        cpu_capacity_(0), gpu_capacity_(0), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  inline size_t size() const { return size_; }
  // set_size sets the "size_" variable.  The current size_ is checked whenever
  // a data accessor/mutator method (cpu_data, gpu_data, mutable_cpu_data, ...)
  // is called and if the current CPU or GPU memory is insufficient to hold the
  // size_, extra space is allocated.  If the current allocation on the device/
  // host (depending on whether cpu_* or gpu_data was called) is sufficient
  // (i.e., capacity >= size_), no action is taken as a result of the set_size
  // call.  Therefore, the actual allocation can only grow, never shrinking
  // (until the SyncedMemory itself is freed/deleted).
  inline void set_size(const size_t size) { size_ = size; }
  inline size_t cpu_capacity() const { return cpu_capacity_; }
  inline size_t gpu_capacity() const { return gpu_capacity_; }

 private:
  void to_cpu();
  void to_gpu();
  size_t cpu_resize();
  size_t gpu_resize();
  void* cpu_data_;
  void* gpu_data_;
  size_t cpu_capacity_;
  size_t gpu_capacity_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
