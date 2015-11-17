#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
#ifdef USE_OCL
    Caffe::cl_state().destroy_buffer(reinterpret_cast<cl_mem>(gpu_ptr_));
#else
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
#endif
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
  {
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
  }
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
  {
#ifdef USE_OCL
    gpu_ptr_ = Caffe::cl_state().create_buffer(CL_MEM_READ_WRITE, size_, NULL);
    ClDeviceProperties cl_prop_version = Caffe::cl_state().get_properties();
    if (cl_prop_version.version == "OpenCL 1.1") {
      caffe_gpu_fill_buffer_size_t(size_, 0, gpu_ptr_);
    } else {
      caffe_gpu_memset(size_, 0, gpu_ptr_);
    }
#else
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
#endif
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  }
  case HEAD_AT_CPU:
  {
    if (gpu_ptr_ == NULL) {
#ifdef USE_OCL
      gpu_ptr_ = Caffe::cl_state().create_buffer(CL_MEM_READ_WRITE, size_,
        NULL);
#else
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
#endif
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
  }
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

inline void SyncedMemory::to_gpu_with_zero_copy(const size_t size,
    void* host_ptr) {
#ifndef CPU_ONLY
#ifdef USE_OCL
  if (host_ptr) {
    ClState& state = Caffe::cl_state();
    switch (head_) {
    case UNINITIALIZED:  // cpu_ptr_ == NULL
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      memcpy(cpu_ptr_, host_ptr, size_);
      own_cpu_data_ = true;
      gpu_ptr_ = state.create_buffer(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
          size_, cpu_ptr_);
      head_ = SYNCED;
      zero_copy_mem_ = true;
      break;
    case HEAD_AT_CPU:    // cpu_ptr_ == host_ptr
      if (gpu_ptr_ == NULL) {
        gpu_ptr_ = state.create_buffer(CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            size_, cpu_ptr_);
        head_ = SYNCED;
        zero_copy_mem_ = true;
      }
      break;
    case SYNCED:
      if (zero_copy_mem_ && cpu_ptr_ != host_ptr) {
        cl_command_queue queue = state.get_command_queue();
        ClMemOff<uint8_t> buf_gpu = state.get_buffer_mem(gpu_ptr_);
        void* mapped_ptr = clEnqueueMapBuffer(queue, buf_gpu.memobj, CL_TRUE,
            CL_MAP_WRITE, buf_gpu.offset, size_, 0, NULL, NULL, NULL);
        memcpy(mapped_ptr, host_ptr, size_);
        clEnqueueUnmapMemObject(queue, buf_gpu.memobj, mapped_ptr, 0, NULL,
            NULL);
      } else {
        // No this case
      }
      break;
    default:
      break;
    }
  }
#endif  // USE_OCL
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

const void* SyncedMemory::gpu_data_with_zero_copy() {
#ifndef CPU_ONLY
  to_gpu_with_zero_copy(size_, cpu_ptr_);
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
#ifndef USE_OCL
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) {
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    }
    CUDA_CHECK(cudaFree(gpu_ptr_));
    cudaSetDevice(initial_device);
#else
    assert(0);
#endif
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void* SyncedMemory::mutable_gpu_data_with_zero_copy(const size_t size,
    void* host_ptr) {
#ifndef CPU_ONLY
  to_gpu_with_zero_copy(size, host_ptr);
  head_ = SYNCED;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

#ifndef CPU_ONLY
#ifndef USE_OCL
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif
#endif

}  // namespace caffe

