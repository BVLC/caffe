// Copyright 2014 BVLC and contributors.
//#ifdef USE_OPENCL
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/opencl_syncedmem.hpp"

namespace caffe {

OpenCLSyncedMemory::~OpenCLSyncedMemory() {
  if (shared_host_ptr_ && this->own_cpu_data_) {
    opencl_aligned_free(shared_host_ptr_);
    shared_host_ptr_ = NULL;
  }

  if (mapped_device_ptr_) {
    CL_CHECK(clReleaseMemObject(device_mem_));
    free(mapped_device_ptr_);
    mapped_device_ptr_ = NULL;
  }
}

inline void OpenCLSyncedMemory::to_cpu() {
  switch (this->head_) {
  case UNINITIALIZED:
    opencl_aligned_malloc(&shared_host_ptr_, &(this->size_));
    memset(shared_host_ptr_, 0, this->size_);
    this->head_ = HEAD_AT_CPU;
    this->own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
    if (shared_host_ptr_ == NULL) {
      opencl_aligned_malloc(&shared_host_ptr_, &(this->size_));
      this->own_cpu_data_ = true;
    }
    CL_CHECK(clEnqueueReadBuffer(
        CaffeOpenCL::queue(), device_mem_, CL_TRUE, 0,
        this->size_, shared_host_ptr_, 0, NULL, NULL));
    this->head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void OpenCLSyncedMemory::to_gpu() {
  switch (this->head_) {
  case UNINITIALIZED:
/*
 * http://streamcomputing.eu/blog/2013-02-03/opencl-basics-flags-for-the-creating-memory-objects/
 */
    opencl_aligned_malloc(&shared_host_ptr_, &(this->size_));
    cl_int error;
    device_mem_ = clCreateBuffer(
        CaffeOpenCL::context(), CL_MEM_USE_HOST_PTR,
        this->size_, shared_host_ptr_, &error);
    CL_CHECK(error);
    this->head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (mapped_device_ptr_ == NULL) {
      cl_int error;
      device_mem_ = clCreateBuffer(
          CaffeOpenCL::context(), CL_MEM_USE_HOST_PTR,
          this->size_, shared_host_ptr_, &error);
      CL_CHECK(error);
      mapped_device_ptr_ = clEnqueueMapBuffer(
          CaffeOpenCL::queue(), device_mem_, CL_TRUE,
          CL_MAP_READ | CL_MAP_WRITE, 0, this->size_, 0, NULL, NULL, &error);
      CL_CHECK(error);
    }
    CL_CHECK(clEnqueueWriteBuffer(
        CaffeOpenCL::queue(), device_mem_, CL_TRUE, 0,
        this->size_, shared_host_ptr_, 0, NULL, NULL));
    this->head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* OpenCLSyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)shared_host_ptr_;
}

void OpenCLSyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (this->own_cpu_data_) {
    CaffeFreeHost(shared_host_ptr_);
  }
  shared_host_ptr_ = data;
  this->head_ = HEAD_AT_CPU;
  this->own_cpu_data_ = false;
}

const void* OpenCLSyncedMemory::gpu_data() {
  to_gpu();
  cl_int error;
  mapped_device_ptr_ = clEnqueueMapBuffer(
      CaffeOpenCL::queue(), device_mem_, CL_TRUE,
      CL_MAP_WRITE, 0, this->size_, 0, NULL, NULL, &error);
  CL_CHECK(error);
  CL_CHECK(clEnqueueUnmapMemObject(
      CaffeOpenCL::queue(), device_mem_, mapped_device_ptr_,
      0, NULL, NULL));
  return (const void*)(mapped_device_ptr_);
}

void* OpenCLSyncedMemory::mutable_cpu_data() {
  to_cpu();
  return shared_host_ptr_;
}

void* OpenCLSyncedMemory::mutable_gpu_data() {
  to_gpu();
  cl_int error;
  mapped_device_ptr_ = clEnqueueMapBuffer(
      CaffeOpenCL::queue(), device_mem_, CL_TRUE,
      CL_MAP_READ | CL_MAP_WRITE, 0, this->size_, 0, NULL, NULL, &error);
  CL_CHECK(error);
  CL_CHECK(clEnqueueUnmapMemObject(
      CaffeOpenCL::queue(), device_mem_, mapped_device_ptr_,
      0, NULL, NULL));
  return mapped_device_ptr_;
}


}  // namespace caffe
//#endif  // USE_OPENCL
