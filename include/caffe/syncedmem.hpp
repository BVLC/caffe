#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

/**
 * SyncedMemory 是caffe中用来管理内存分配和CPU、GPU数据及同步的类，只服务于Blob类
 * 目的是为了屏蔽上层代码对不同硬件设备的内存分配的感知，同时隐藏了CPU和GPU之间的同步过程。
 * 同时，SyncedMemory实现时，采用的是 “lazy”的模式，就是内存的实际申请时机是在第一次使用时进行的。
 * 
*/
namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
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
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu(); //CPU数据指针
  void to_gpu(); //GPU数据指针
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_; //当前 SyncedMemory需要维护的数据个数
  /**
   * 当第一次调用to_cpu(),head_处于UNINITIALIZED状态，那么系统会调用CPU的申请内存的方式去获得内存区域，之后设置 head_ = HEAD_AT_CPU,
   * 如果中间过程没有GPU设备则不会有状态变动，如果中间有代码调用了 to_gpu() ,则会发现 head_处于 HEAD_AT_CPU 状态，此时会调用同步函数，将数据从CPU同步到GPU,
   * 之后如果又回到CPU上，则同样会发现 head_ 处于HEAD_AT_GPU的状态，那么又会调用相应的同步代码，将数据同步回CPU，
   * 通过 head_这样一个状态参数屏蔽了GPU和CPU间的申请和切换的不同。
  */
  SyncedHead head_; //当前 SyncedMemory处于的状态
  // *own_cpu_data_和own_gpu_data_这两个变量，这两个变量主要是用来记录是否使用了共享的数据还是自己的数据
  bool own_cpu_data_; 
  bool cpu_malloc_use_cuda_; //是否使用cuda
  bool own_gpu_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
