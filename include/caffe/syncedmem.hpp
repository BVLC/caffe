#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#include "caffe/common.hpp"

namespace caffe {

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  *ptr = malloc(size);
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
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
inline void CaffeMallocHost(void** ptr, size_t size) {
#ifndef CPU_ONLY
  cudaMallocHost(ptr, size);
#else
  *ptr = malloc(size);
#endif
}

inline void CaffeFreeHost(void* ptr) {
#ifndef CPU_ONLY
  cudaFreeHost(ptr);
#else
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  *ptr = malloc(size);
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
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  free(ptr);
#endif
}

<<<<<<< HEAD
=======
<<<<<<< HEAD

<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/caffe-merge
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
inline void CaffeFreeHost(void* ptr, bool use_cuda) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
>>>>>>> pod/common.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> device-abstraction
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
=======
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
>>>>>>> device-abstraction
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
  free(ptr);
#endif
}
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
<<<<<<< HEAD
=======

<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD

<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======

<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> pod/common.hpp
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
=======
>>>>>>> origin/BVLC/parallel
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
<<<<<<< HEAD
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
        own_cpu_data_(false), own_gpu_data_(false) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), own_gpu_data_(false) {}
>>>>>>> origin/BVLC/parallel
=======
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
  explicit SyncedMemory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {}
>>>>>>> device-abstraction
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
=======
<<<<<<< HEAD
=======
  const void* const_data();
  void* mutable_data();
=======
>>>>>>> pod/caffe-merge
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  const void* const_data();
  void* mutable_data();
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
>>>>>>> BVLC/master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> device-abstraction
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/caffe-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> master
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod-caffe-pod.hpp-merge
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
<<<<<<< HEAD
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> pod/common.hpp
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
=======
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> BVLC/master
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> master
>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction
=======
  const void* const_data();
  void* mutable_data();
=======
#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
  const void* const_data();
  void* mutable_data();
>>>>>>> BVLC/device-abstraction

 private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
<<<<<<< HEAD
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  bool own_gpu_data_;
>>>>>>> origin/BVLC/parallel
=======
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;
>>>>>>> device-abstraction

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
