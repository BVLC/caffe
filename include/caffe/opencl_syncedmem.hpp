// Copyright 2014 BVLC and contributors.
//#ifdef USE_OPENCL
#ifndef CAFFE_OPENCL_SYNCEDMEM_HPP_
#define CAFFE_OPENCL_SYNCEDMEM_HPP_

#include <cstdlib>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/opencl_device.hpp"

namespace caffe {


/*
 * https://software.intel.com/sites/products/documentation/ioclsdk/2013/OG/Mapping_Memory_Objects_(USE_HOST_PTR).htm
 * For efficiency reasons such a host-side pointer must be allocated for the
 *   conditions:
 * * The amount of memory you allocate and the size of the corresponding
 * *   OpenCL* buffer must be multiple of the cache line sizes (64 bytes).
 * * Always use 4k alignment (page alignment) when you allocate the host memory
 * *   for sharing with OpenCL devices.
 */
#define OPENCL_CACHE_LINE_SIZE 64
#define OPENCL_PAGE_ALIGNMENT 4096

inline void opencl_aligned_malloc(void** ptr, size_t* size) {
  *size +=  (*size % OPENCL_CACHE_LINE_SIZE);
#ifdef _MSC_VER
  *ptr = _aligned_malloc(*size, OPENCL_PAGE_ALIGNMENT);
#else
  if(posix_memalign(ptr, OPENCL_PAGE_ALIGNMENT, *size)) {
    *ptr = NULL;
  }
#endif
}

inline void opencl_aligned_free(void* ptr) {
#ifdef _MSC_VER
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

class OpenCLSyncedMemory {
 public:
  OpenCLSyncedMemory()
      : shared_host_ptr_(NULL), mapped_device_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  explicit OpenCLSyncedMemory(size_t size)
      : shared_host_ptr_(NULL), mapped_device_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false) {}
  ~OpenCLSyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

 private:
  void to_cpu();
  void to_gpu();
  void* shared_host_ptr_;
  void* mapped_device_ptr_;
  cl_mem device_mem_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;

  DISABLE_COPY_AND_ASSIGN(OpenCLSyncedMemory);
};  // class OpenCLSyncedMemory

}  // namespace caffe

#endif  // CAFFE_OPENCL_SYNCEDMEM_HPP_
//#endif  // USE_OPENCL
