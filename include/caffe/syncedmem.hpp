#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"
#include "caffe/backend/vptr.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

class Device;

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 */

class SyncedMemory {
 public:
  explicit SyncedMemory(Device *device_context)
      : cpu_ptr_(NULL),
        gpu_ptr_(vptr<void>()),
        size_(0),
        head_(UNINITIALIZED),
        own_cpu_data_(false),
        own_gpu_data_(false),
        own_zero_copy_data_(false),
        device_(device_context) {
  }
  explicit SyncedMemory(uint_tp size, Device *device_context)
      : cpu_ptr_(NULL),
        gpu_ptr_(vptr<void>()),
        size_(size),
        head_(UNINITIALIZED),
        own_cpu_data_(false),
        own_gpu_data_(false),
        own_zero_copy_data_(false),
        device_(device_context) {
  }

  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  vptr<const void> gpu_data();
  void set_gpu_data(vptr<void> data);
  void* mutable_cpu_data();
  vptr<void> mutable_gpu_data();
  enum SyncedHead {
    UNINITIALIZED,
    HEAD_AT_CPU,
    HEAD_AT_GPU,
    SYNCED
  };
  SyncedHead head() const {
    return head_;
  }
  uint_tp size() const {
    return size_;
  }

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  vptr<void> gpu_ptr_;

  uint_tp size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool own_gpu_data_;
  bool own_zero_copy_data_;
  Device *device_;

DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};
// class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
