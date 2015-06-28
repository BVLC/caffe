#include <cstring>

#include "caffe/common.hpp"
#include "caffe/device_context.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/greentea/greentea.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
      CUDA_CHECK(cudaFree(gpu_ptr_));
      device_context_->DecreaseMemoryUsage(size_);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      clReleaseMemObject(cl_gpu_mem_);
      device_context_->DecreaseMemoryUsage(size_);
#endif  // USE_GREENTEA
    }
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
    case UNINITIALIZED: {
      CaffeMallocHost(&cpu_ptr_, size_);
      caffe_memset(size_, 0, cpu_ptr_);
      head_ = HEAD_AT_CPU;
      own_cpu_data_ = true;
      break;
    }
    case HEAD_AT_GPU: {
#ifndef CPU_ONLY
      if (cpu_ptr_ == NULL) {
        CaffeMallocHost(&cpu_ptr_, size_);
        own_cpu_data_ = true;
      }
      if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
        caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context ctx = viennacl::ocl::get_context(
            device_context_->id());
        ctx.get_queue().finish();
        // On the CPU, memory is shared (and no copy needed)
        if (ctx.devices()[0].type() != CL_DEVICE_TYPE_CPU) {
          greentea_gpu_memcpy(size_, (cl_mem) gpu_ptr_, 0, cpu_ptr_, &ctx);
        }
        ctx.get_queue().finish();
#endif
      }
      head_ = SYNCED;
#else
      NO_GPU;
#endif
      break;
    }
    case HEAD_AT_CPU:
    case SYNCED:
      break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
    case UNINITIALIZED: {
      if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
        CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
        device_context_->IncreaseMemoryUsage(size_);
        caffe_gpu_memset(size_, 0, gpu_ptr_);
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context ctx = viennacl::ocl::get_context(
            device_context_->id());
        ctx.get_queue().finish();
        cl_int err;
        if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
          // CPU memory is shared
          if (cpu_ptr_ == NULL) {
            CaffeMallocHost(&cpu_ptr_, size_);
            caffe_memset(size_, 0, cpu_ptr_);
          }
          cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(),
          CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                       size_, cpu_ptr_, &err);
          device_context_->IncreaseMemoryUsage(size_);
        } else {
          cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                       size_, NULL, &err);
          device_context_->IncreaseMemoryUsage(size_);
          int alpha = 0;
          greentea_memset(device_context_->id(), size_, alpha, cl_gpu_mem_, 0);
        }
        gpu_ptr_ = reinterpret_cast<void*>(cl_gpu_mem_);
        ctx.get_queue().finish();
#endif
      }
      head_ = HEAD_AT_GPU;
      break;
    }
    case HEAD_AT_CPU: {
      if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
        if (gpu_ptr_ == NULL) {
          CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
          device_context_->IncreaseMemoryUsage(size_);
        }
        caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
#endif  // USE_CUDA
      } else {
#ifdef USE_GREENTEA
        viennacl::ocl::context ctx = viennacl::ocl::get_context(
            device_context_->id());
        ctx.get_queue().finish();
        if (gpu_ptr_ == NULL) {
          cl_int err;
          if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
            // CPU memory is shared
            if (cpu_ptr_ == NULL) {
              CaffeMallocHost(&cpu_ptr_, size_);
            }
            cl_gpu_mem_ = clCreateBuffer(
                ctx.handle().get(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                size_, cpu_ptr_, &err);
            device_context_->IncreaseMemoryUsage(size_);

          } else {
            cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                         size_, NULL, &err);
            device_context_->IncreaseMemoryUsage(size_);
          }
          gpu_ptr_ = reinterpret_cast<void*>(cl_gpu_mem_);
          ctx.get_queue().finish();
        }
        // On the CPU, memory is shared (and no copy needed)
        if (ctx.devices()[0].type() != CL_DEVICE_TYPE_CPU) {
          greentea_gpu_memcpy(size_, cpu_ptr_, (cl_mem) gpu_ptr_, 0, &ctx);
        }
        ctx.get_queue().finish();
#endif  // USE_GREENTEA
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
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  if (device_context_->backend() == Backend::BACKEND_OpenCL) {
#ifdef USE_GREENTEA
    viennacl::ocl::context ctx = viennacl::ocl::get_context(
        device_context_->id());
    if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
      // If host memory is released and shared
      gpu_ptr_ = NULL;
    }
#endif
  }
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*) gpu_ptr_;
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
#endif
}

}  // namespace caffe

