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
}

inline void CaffeFreeHost(void* ptr) {
  free(ptr);
}

class AbstractSyncedMemory {
 public:
  AbstractSyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  explicit AbstractSyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  virtual ~AbstractSyncedMemory() {}
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  virtual const void* cpu_data() = 0;
  virtual void set_cpu_data(void* data) = 0;
  virtual const void* gpu_data() = 0;
  virtual void* mutable_cpu_data() = 0;
  virtual void* mutable_gpu_data() = 0;
  virtual SyncedHead head() { return head_; }
  virtual size_t size() { return size_; }

  const void* const_data() const { return NULL; }
  void* mutable_data() { return NULL;}

 protected:
  virtual void to_cpu() = 0;
  virtual void to_gpu() = 0;

 protected:
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
};

class SyncedMemory : public AbstractSyncedMemory {
 public:
  SyncedMemory() : AbstractSyncedMemory() {}
  explicit SyncedMemory(size_t size) : AbstractSyncedMemory(size) {}
  virtual ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();

 protected:
  void to_cpu();
  void to_gpu();

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
