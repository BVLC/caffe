#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include "caffe/syncedmem.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
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
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
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
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
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
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
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
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
=======
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
>>>>>>> BVLC/device-abstraction
=======
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
<<<<<<< HEAD
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
>>>>>>> BVLC/device-abstraction
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
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
>>>>>>> BVLC/device-abstraction
=======
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
=======
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
>>>>>>> BVLC/master
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
    CaffeMallocHost(&cpu_ptr_, size_);
    GetDevice<float>(Caffe::CPU)->set_void(size_, 0, cpu_ptr_);
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> caffe
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    GetDevice<float>(Caffe::GPU)->copy_void(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    GetDevice<float>(Caffe::GPU)->set_void(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    GetDevice<float>(Caffe::GPU)->copy_void(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
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
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
#endif
=======
=======
=======
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
>>>>>>> pod/common.hpp
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
void SyncedMemory::set_gpu_data(void* data) {
=======
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
=======
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
=======
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
<<<<<<< HEAD
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));
>>>>>>> pod/common.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
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
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
>>>>>>> pod-caffe-pod.hpp-merge
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel

>>>>>>> pod/common.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
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
const void* SyncedMemory::const_data() {
=======
=======

>>>>>>> BVLC/device-abstraction
void* SyncedMemory::mutable_data() {
>>>>>>> pod/device/blob.hpp
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
>>>>>>> pod/common.hpp
  }
<<<<<<< HEAD
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
=======
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
<<<<<<< HEAD
=======
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
>>>>>>> pod/device/blob.hpp
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
=======
<<<<<<< HEAD
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
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
>>>>>>> pod/device/blob.hpp
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
#endif
}

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
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> device-abstraction
=======
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
<<<<<<< HEAD
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
>>>>>>> BVLC/master
=======
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
>>>>>>> BVLC/device-abstraction

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> device-abstraction
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
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
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
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
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
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
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));
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
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/common.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
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
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> pod/common.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/device-abstraction
=======
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
>>>>>>> origin/BVLC/parallel

<<<<<<< HEAD
void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> BVLC/device-abstraction
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
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
>>>>>>> pod-caffe-pod.hpp-merge
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp

<<<<<<< HEAD
=======

>>>>>>> BVLC/device-abstraction
void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
=======
<<<<<<< HEAD
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod/common.hpp

<<<<<<< HEAD
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
const void* SyncedMemory::const_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return cpu_data();
  case Caffe::GPU:
    return gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD

>>>>>>> pod-caffe-pod.hpp-merge
=======

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/common.hpp
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
=======
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> origin/BVLC/parallel
>>>>>>> pod/device/blob.hpp
=======
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
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
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
>>>>>>> pod/common.hpp
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe

>>>>>>> pod/caffe-merge
=======

void* SyncedMemory::mutable_data() {
=======
=======
=======
>>>>>>> BVLC/device-abstraction
const void* SyncedMemory::const_data() {
>>>>>>> pod/device/blob.hpp
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
<<<<<<< HEAD
    return mutable_gpu_data();
=======
    return gpu_data();
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
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction

=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe
=======
>>>>>>> origin/BVLC/parallel
<<<<<<< HEAD
>>>>>>> pod/common.hpp

<<<<<<< HEAD
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======

>>>>>>> BVLC/device-abstraction
=======
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
>>>>>>> BVLC/master

=======
=======
>>>>>>> origin/BVLC/parallel

<<<<<<< HEAD
=======

>>>>>>> BVLC/device-abstraction
>>>>>>> pod/device/blob.hpp
void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD

>>>>>>> device-abstraction
=======
=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
<<<<<<< HEAD
=======
>>>>>>> BVLC/device-abstraction
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> origin/BVLC/parallel

<<<<<<< HEAD
=======

>>>>>>> BVLC/device-abstraction
void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice, stream));
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
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
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> caffe

>>>>>>> pod/caffe-merge
=======
>>>>>>> BVLC/device-abstraction

=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> BVLC/master

void* SyncedMemory::mutable_data() {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    return mutable_cpu_data();
  case Caffe::GPU:
    return mutable_gpu_data();
  default:
    LOG(FATAL) << "Unknown caffe mode.";
    return static_cast<void*>(0);
  }
}
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> BVLC/device-abstraction

=======
=======
>>>>>>> BVLC/master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> caffe
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> BVLC/master
=======
>>>>>>> master
=======
>>>>>>> master
=======
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> caffe

>>>>>>> pod-caffe-pod.hpp-merge
=======

>>>>>>> BVLC/device-abstraction
}  // namespace caffe
