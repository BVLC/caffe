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
#ifndef CPU_ONLY
  if (gpu_ptr_) {
    if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
      // Free device memory
      cudaFree(gpu_ptr_);
      gpu_ptr_ = nullptr;
      device_context_->DecreaseMemoryUsage(size_);
#endif  // USE_CUDA
    } else {
#ifdef USE_GREENTEA
      // Free device memory
      viennacl::ocl::context ctx = viennacl::ocl::get_context(
          device_context_->id());
      ctx.get_queue().finish();
      CHECK_EQ(CL_SUCCESS, clReleaseMemObject(cl_gpu_mem_))
          << "OpenCL memory corruption";
      gpu_ptr_ = nullptr;
      cl_gpu_mem_ = nullptr;
      ctx.get_queue().finish();
      device_context_->DecreaseMemoryUsage(size_);
#endif  // USE_GREENTEA
    }
  }
#endif  // !CPU_ONLY
  // Free host memory
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
    cpu_ptr_ = nullptr;
  }
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
      if (cpu_ptr_ == nullptr) {
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
        greentea_gpu_memcpy(size_, (cl_mem) gpu_ptr_, 0, cpu_ptr_, &ctx);
        ctx.get_queue().finish();
#endif
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
          cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(),
                     CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                     size_, nullptr, &err);
        } else {
          cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                       size_, nullptr, &err);
        }
        CHECK_EQ(0, err) << "OpenCL buffer allocation of size "
                        << size_ << " failed.";
        device_context_->IncreaseMemoryUsage(size_);
        int alpha = 0;
        greentea_memset(device_context_->id(), size_, alpha, cl_gpu_mem_, 0);
        gpu_ptr_ = reinterpret_cast<void*>(cl_gpu_mem_);
        ctx.get_queue().finish();
#endif  // USE_GREENTEA
      }
      head_ = HEAD_AT_GPU;
      break;
    }
    case HEAD_AT_CPU: {
      if (device_context_->backend() == Backend::BACKEND_CUDA) {
#ifdef USE_CUDA
        if (gpu_ptr_ == nullptr) {
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
        if (gpu_ptr_ == nullptr) {
          cl_int err;
          if (ctx.devices()[0].type() == CL_DEVICE_TYPE_CPU) {
            cl_gpu_mem_ = clCreateBuffer(
                ctx.handle().get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                size_, nullptr, &err);
          } else {
            cl_gpu_mem_ = clCreateBuffer(ctx.handle().get(), CL_MEM_READ_WRITE,
                                         size_, nullptr, &err);
          }
          CHECK_EQ(0, err) << "OpenCL buffer allocation of size "
                          << size_ << " failed.";
          device_context_->IncreaseMemoryUsage(size_);
          gpu_ptr_ = reinterpret_cast<void*>(cl_gpu_mem_);
          ctx.get_queue().finish();
        }
        greentea_gpu_memcpy(size_, cpu_ptr_, (cl_mem) gpu_ptr_, 0, &ctx);
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
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
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

