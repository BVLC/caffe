#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"

#include "caffe/backend/device.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
#ifndef CPU_ONLY
  // Free device memory
  if (gpu_ptr_.is_valid() && own_gpu_data_) {
    device_->FreeMemDevice(gpu_ptr_);
    device_->decrease_memory_usage(size_);
    gpu_ptr_ = vptr<void>();
  }
#endif  // !CPU_ONLY
  // Free host memory
  if (cpu_ptr_ && (own_cpu_data_ || own_zero_copy_data_)) {
    device_->FreeMemHost(cpu_ptr_);
    cpu_ptr_ = nullptr;
  }
}

inline void SyncedMemory::to_cpu() {
  if (size_ == 0)
    LOG(FATAL) << "Trying to access memory of size 0.";

  switch (head_) {
    case UNINITIALIZED: {
#ifndef CPU_ONLY
      if (device_->backend() != BACKEND_CPU && device_->is_host_unified()) {
        gpu_ptr_ = device_->MallocMemDevice(size_, &cpu_ptr_,
                                            device_->is_host_unified());
        device_->increase_memory_usage(size_);
        device_->memset(size_, 0, gpu_ptr_);
        device_->FinishQueues();  // Required to synchronize CPU-GPU states
        own_gpu_data_ = true;
        own_cpu_data_ = true;
        own_zero_copy_data_ = true;
        head_ = SYNCED;
      } else {
#endif  // !CPU_ONLY
        device_->MallocMemHost(size_, &cpu_ptr_);
        caffe_memset(size_, 0, cpu_ptr_);
        own_cpu_data_ = true;
        head_ = HEAD_AT_CPU;
#ifndef CPU_ONLY
      }
#endif  // !CPU_ONLY
      break;
    }
    case HEAD_AT_GPU: {
#ifndef CPU_ONLY
      if (cpu_ptr_ == nullptr) {
        device_->MallocMemHost(size_, &cpu_ptr_);
        own_cpu_data_ = true;
      }
      if (own_zero_copy_data_) {
        device_->CheckZeroCopy(gpu_ptr_, cpu_ptr_, size_);
      } else {
        device_->memcpy(size_, gpu_ptr_, cpu_ptr_);
      }
      head_ = SYNCED;
#else
      NO_GPU;
#endif  // !CPU_ONLY
      break;
    }
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

inline void SyncedMemory::to_gpu() {
  if (size_ == 0)
    LOG(FATAL) << "Trying to access memory of size 0.";

#ifndef CPU_ONLY
  switch (head_) {
    case UNINITIALIZED: {
      gpu_ptr_ = device_->MallocMemDevice(size_, &cpu_ptr_,
                                          device_->is_host_unified());
      device_->increase_memory_usage(size_);
      device_->memset(size_, 0, gpu_ptr_);
      own_gpu_data_ = true;
      if (device_->is_host_unified()) {
        device_->FinishQueues();  // Required to synchronize CPU-GPU states
        own_cpu_data_ = true;
        own_zero_copy_data_ = true;
        head_ = SYNCED;
      } else {
        head_ = HEAD_AT_GPU;
      }
      break;
    }
    case HEAD_AT_CPU: {
      if (gpu_ptr_.get() == nullptr) {
        gpu_ptr_ = device_->MallocMemDevice(size_, &cpu_ptr_, false);
        device_->increase_memory_usage(size_);
        own_gpu_data_ = true;
        own_zero_copy_data_ = false;
      }
      if (own_zero_copy_data_) {
        device_->CheckZeroCopy(gpu_ptr_, cpu_ptr_, size_);
      } else {
        device_->memcpy(size_, cpu_ptr_, gpu_ptr_);
      }
      head_ = SYNCED;
      break;
    }
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
  return (const void*) cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (cpu_ptr_ && own_cpu_data_) {
    device_->FreeMemHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
  if (own_zero_copy_data_) {
    gpu_ptr_ = vptr<void>();
    own_gpu_data_ = false;
    own_zero_copy_data_ = false;
  }
}

vptr<const void> SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(vptr<void> data) {
#ifndef CPU_ONLY
  if (own_gpu_data_) {
    device_->FreeMemDevice(gpu_ptr_);
    if (own_zero_copy_data_) {
      device_->FreeMemHost(cpu_ptr_);
    }
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
  if (own_zero_copy_data_) {
    cpu_ptr_ = nullptr;
    own_cpu_data_ = false;
    own_zero_copy_data_ = false;
  }
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

vptr<void> SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

// TODO: Implement this function device abstracted
/*
#ifndef CPU_ONLY
#ifdef USE_CUDA
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif  // USE_CUDA
#endif  // !CPU_ONLY
*/

}  // namespace caffe

