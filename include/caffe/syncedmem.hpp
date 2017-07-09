#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
// 进行内存分配
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

// 释放内存
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *        在主机（CPU）与设备（GPU）之间进行内存的分配及同步
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  // 获得CPU端的数据（不可变），通常用来获取bottom的数据
  const void* cpu_data();
  void set_cpu_data(void* data);
  // 获得GPU端的数据（不可变），通常用来获取bottom的数据
  const void* gpu_data();
  void set_gpu_data(void* data);
  // 获得CPU端的数据（可变），通常用于top数据的计算
  void* mutable_cpu_data();
  // 获得GPU端的数据（可变），通常用于top数据的计算
  void* mutable_gpu_data();
  // 同步信息（未初始化，在CPU端，在GPU端，已在CPU和GPU之间进行同步）
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  // 检查设备
  void check_device();
  // 将数据同步到CPU
  void to_cpu();
  // 将数据同步到GPU
  void to_gpu();
  // CPU端数据的指针
  void* cpu_ptr_;
  // GPU端数据的指针
  void* gpu_ptr_;
  // 数据的大小
  size_t size_;
  // 数据同步的状态
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int device_; // 设备号

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
