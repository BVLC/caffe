#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
#include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void *cpu_data();
  void set_cpu_data(void *data);
  const void *gpu_data();
  void set_gpu_data(void *data);
  void *mutable_cpu_data();
  void *mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

  static size_t get_used_size();
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t &stream);
#endif

  static void *gpu_malloc(size_t size);
  static void gpu_free(void *data);

private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void *cpu_ptr_;
  void *gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
}; // class SyncedMemory

} // namespace caffe

#endif // CAFFE_SYNCEDMEM_HPP_
