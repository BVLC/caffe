/*
 * device_context.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: Fabian Tschopp
 */

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/device.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/device_alternate.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/cl_kernels.hpp"
#endif  // USE_GREENTEA

namespace caffe {

device::device()
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(0), list_id_(0),
      backend_(Backend::BACKEND_CPU), memory_usage_(0), peak_memory_usage_(0),
      host_unified_(false) {
}

device::device(int id, int list_id, Backend backend)
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(id), list_id_(list_id),
      backend_(backend), memory_usage_(0), peak_memory_usage_(0),
      host_unified_(false) {
}

void device::Init() {
#ifndef CPU_ONLY
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    workgroup_sizes_[0] = CAFFE_CUDA_NUM_THREADS;
    workgroup_sizes_[1] = CAFFE_CUDA_NUM_THREADS;
    workgroup_sizes_[2] = CAFFE_CUDA_NUM_THREADS;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);

    std::vector<size_t> temp(3);
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_MAX_WORK_ITEM_SIZES,
                    3 * sizeof(size_t), &temp[0], NULL);
    workgroup_sizes_[0] = temp[0];
    workgroup_sizes_[1] = temp[1];
    workgroup_sizes_[2] = temp[2];
    cl_bool host_unified;
    clGetDeviceInfo(ctx.devices()[0].id(),
                    CL_DEVICE_HOST_UNIFIED_MEMORY,
                    sizeof(cl_bool), &host_unified, NULL);

    host_unified_ = host_unified;
    SetProgram();

    for (int q = 0; q < GREENTEA_QUEUE_COUNT - 1; ++q) {
      ctx.add_queue(ctx.devices()[0]);
    }
#endif  // USE_GREENTEA
  }
#endif  // !CPU_ONLY
}

Backend device::backend() const {
  return backend_;
}

int device::id() const {
  return id_;
}

int device::list_id() const {
  return list_id_;
}

int device::workgroup_size(int id) {
  return workgroup_sizes_[id % 3];
}

int device::num_queues() {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    return 1;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    return GREENTEA_QUEUE_COUNT;
#endif  // USE_GREENTEA
  }
  return 1;
}

template<>
shared_ptr<Blob<float> > device::Buffer(int id) {
  if (buff_f_.size() <= id) {
    shared_ptr<Blob<float> > blob_pointer(new Blob<float>(this));
    buff_f_.push_back(blob_pointer);
  }
  return buff_f_[id];
}

template<>
shared_ptr<Blob<double> > device::Buffer(int id) {
  if (buff_d_.size() <= id) {
    shared_ptr<Blob<double> > blob_pointer(new Blob<double>(this));
    buff_d_.push_back(blob_pointer);
  }
  return buff_d_[id];
}

int device::current_queue_id() {
  return current_queue_id_;
}

void device::SwitchQueue(int id) {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    (void) id;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
    ctx.switch_queue(id % num_queues());
    current_queue_id_ = id % num_queues();
#endif  // USE_GREENTEA
  }
}

void device::FinishQueues() {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
    for (int i = 0; i < num_queues(); ++i) {
      ctx.switch_queue(i);
      ctx.get_queue().finish();
    }
    ctx.switch_queue(0);
    current_queue_id_ = 0;
#endif  // USE_GREENTEA
  }
}

uint_tp device::memory_usage() {
  return memory_usage_;
}

uint_tp device::peak_memory_usage() {
  return peak_memory_usage_;
}

void device::IncreaseMemoryUsage(uint_tp bytes) {
  memory_usage_ += bytes;
  if (memory_usage_ > peak_memory_usage_) {
    peak_memory_usage_ = memory_usage_;
  }
}

void device::DecreaseMemoryUsage(uint_tp bytes) {
  memory_usage_ -= bytes;
}

void device::ResetPeakMemoryUsage() {
  peak_memory_usage_ = memory_usage_;
}

bool device::CheckCapability(std::string cap) {
  if (backend_ == BACKEND_OpenCL) {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);

    size_t size;
    size_t max_size = 1024 * 1024;
    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_EXTENSIONS,
                    0, NULL, &size);

    // Cap at 1 MB to capture faulty OpenCL implementations (nVidia)
    std::vector<char> exts(std::min(size, max_size));

    clGetDeviceInfo(ctx.devices()[0].id(), CL_DEVICE_EXTENSIONS,
                    size, &(exts[0]), NULL);

    std::string extsstr(&(exts[0]));
    return extsstr.find(cap) != std::string::npos;
#endif
  }
  return false;
}

bool device::CheckVendor(std::string vendor) {
  if (backend_ == BACKEND_CUDA) {
    if (vendor.compare("NVIDIA") == 0)
      return true;
  }
#ifdef USE_GREENTEA
  else if (backend_ == BACKEND_OpenCL) {
    viennacl::ocl::context &ctx = viennacl::ocl::get_context(id_);
    const viennacl::ocl::device &device = ctx.current_device();

    if (device.vendor().find(vendor) != std::string::npos)
      return true;
  }
#endif

  return false;
}

#ifdef USE_GREENTEA
viennacl::ocl::program &device::program() {
  return ocl_program_;
}

void device::SetProgram() {
  ocl_program_ = RegisterKernels(
      &(viennacl::ocl::get_context(static_cast<uint64_t>(id_))));
}

bool device::is_host_unified() {
  return host_unified_;
}

const char* clGetErrorString(cl_int error) {
  switch (error) {
  case 0: return "CL_SUCCESS";
  case -1: return "CL_DEVICE_NOT_FOUND";
  case -2: return "CL_DEVICE_NOT_AVAILABLE";
  case -3: return "CL_COMPILER_NOT_AVAILABLE";
  case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
  case -5: return "CL_OUT_OF_RESOURCES";
  case -6: return "CL_OUT_OF_HOST_MEMORY";
  case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
  case -8: return "CL_MEM_COPY_OVERLAP";
  case -9: return "CL_IMAGE_FORMAT_MISMATCH";
  case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
  case -11: return "CL_BUILD_PROGRAM_FAILURE";
  case -12: return "CL_MAP_FAILURE";
  case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
  case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
  case -15: return "CL_COMPILE_PROGRAM_FAILURE";
  case -16: return "CL_LINKER_NOT_AVAILABLE";
  case -17: return "CL_LINK_PROGRAM_FAILURE";
  case -18: return "CL_DEVICE_PARTITION_FAILED";
  case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
  case -30: return "CL_INVALID_VALUE";
  case -31: return "CL_INVALID_DEVICE_TYPE";
  case -32: return "CL_INVALID_PLATFORM";
  case -33: return "CL_INVALID_DEVICE";
  case -34: return "CL_INVALID_CONTEXT";
  case -35: return "CL_INVALID_QUEUE_PROPERTIES";
  case -36: return "CL_INVALID_COMMAND_QUEUE";
  case -37: return "CL_INVALID_HOST_PTR";
  case -38: return "CL_INVALID_MEM_OBJECT";
  case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case -40: return "CL_INVALID_IMAGE_SIZE";
  case -41: return "CL_INVALID_SAMPLER";
  case -42: return "CL_INVALID_BINARY";
  case -43: return "CL_INVALID_BUILD_OPTIONS";
  case -44: return "CL_INVALID_PROGRAM";
  case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
  case -46: return "CL_INVALID_KERNEL_NAME";
  case -47: return "CL_INVALID_KERNEL_DEFINITION";
  case -48: return "CL_INVALID_KERNEL";
  case -49: return "CL_INVALID_ARG_INDEX";
  case -50: return "CL_INVALID_ARG_VALUE";
  case -51: return "CL_INVALID_ARG_SIZE";
  case -52: return "CL_INVALID_KERNEL_ARGS";
  case -53: return "CL_INVALID_WORK_DIMENSION";
  case -54: return "CL_INVALID_WORK_GROUP_SIZE";
  case -55: return "CL_INVALID_WORK_ITEM_SIZE";
  case -56: return "CL_INVALID_GLOBAL_OFFSET";
  case -57: return "CL_INVALID_EVENT_WAIT_LIST";
  case -58: return "CL_INVALID_EVENT";
  case -59: return "CL_INVALID_OPERATION";
  case -60: return "CL_INVALID_GL_OBJECT";
  case -61: return "CL_INVALID_BUFFER_SIZE";
  case -62: return "CL_INVALID_MIP_LEVEL";
  case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
  case -64: return "CL_INVALID_PROPERTY";
  case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
  case -66: return "CL_INVALID_COMPILER_OPTIONS";
  case -67: return "CL_INVALID_LINKER_OPTIONS";
  case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
  case -69: return "CL_INVALID_PIPE_SIZE";
  case -70: return "CL_INVALID_DEVICE_QUEUE";
  case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
  case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
  case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
  case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
  case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
  case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
  case -1024: return "clBLAS: Functionality is not implemented";
  case -1023: return "clBLAS: Library is not initialized yet";
  case -1022: return "clBLAS: Matrix A is not a valid memory object";
  case -1021: return "clBLAS: Matrix B is not a valid memory object";
  case -1020: return "clBLAS: Matrix C is not a valid memory object";
  case -1019: return "clBLAS: Vector X is not a valid memory object";
  case -1018: return "clBLAS: Vector Y is not a valid memory object";
  case -1017: return "clBLAS: An input dimension (M:N:K) is invalid";
  case -1016: return "clBLAS: Leading dimension A must not be less than the "
      "size of the first dimension";
  case -1015: return "clBLAS: Leading dimension B must not be less than the "
      "size of the second dimension";
  case -1014: return "clBLAS: Leading dimension C must not be less than the "
      "size of the third dimension";
  case -1013: return "clBLAS: The increment for a vector X must not be 0";
  case -1012: return "clBLAS: The increment for a vector Y must not be 0";
  case -1011: return "clBLAS: The memory object for Matrix A is too small";
  case -1010: return "clBLAS: The memory object for Matrix B is too small";
  case -1009: return "clBLAS: The memory object for Matrix C is too small";
  case -1008: return "clBLAS: The memory object for Vector X is too small";
  case -1007: return "clBLAS: The memory object for Vector Y is too small";
  default: return "Unknown OpenCL error";
  }
}

#ifdef USE_FFT
const char* clfftGetErrorString(clfftStatus status) {
  switch (status) {
  case CLFFT_SUCCESS:
    return "CLFFT_SUCCESS";
  case CLFFT_INVALID_PLAN:
    return "CLFFT_INVALID_PLAN";
  case CLFFT_INVALID_GLOBAL_WORK_SIZE:
    return "CLFFT_INVALID_GLOBAL_WORK_SIZE";
  case CLFFT_INVALID_MIP_LEVEL:
    return "CLFFT_INVALID_MIP_LEVEL";
  case CLFFT_INVALID_BUFFER_SIZE:
    return "CLFFT_INVALID_BUFFER_SIZE";
  case CLFFT_INVALID_GL_OBJECT:
    return "CLFFT_INVALID_GL_OBJECT";
  case CLFFT_INVALID_OPERATION:
    return "CLFFT_INVALID_OPERATION";
  case CLFFT_INVALID_EVENT:
    return "CLFFT_INVALID_EVENT";
  case CLFFT_INVALID_EVENT_WAIT_LIST:
    return "CLFFT_INVALID_EVENT_WAIT_LIST";
  case CLFFT_INVALID_GLOBAL_OFFSET:
    return "CLFFT_INVALID_GLOBAL_OFFSET";
  case CLFFT_INVALID_WORK_ITEM_SIZE:
    return "CLFFT_INVALID_WORK_ITEM_SIZE";
  case CLFFT_INVALID_WORK_GROUP_SIZE:
    return "CLFFT_INVALID_WORK_GROUP_SIZE";
  case CLFFT_INVALID_WORK_DIMENSION:
    return "CLFFT_INVALID_WORK_DIMENSION";
  case CLFFT_INVALID_KERNEL_ARGS:
    return "CLFFT_INVALID_KERNEL_ARGS";
  case CLFFT_INVALID_ARG_SIZE:
    return "CLFFT_INVALID_ARG_SIZE";
  case CLFFT_INVALID_ARG_VALUE:
    return "CLFFT_INVALID_ARG_VALUE";
  case CLFFT_INVALID_ARG_INDEX:
    return "CLFFT_INVALID_ARG_INDEX";
  case CLFFT_INVALID_KERNEL:
    return "CLFFT_INVALID_KERNEL";
  case CLFFT_INVALID_KERNEL_DEFINITION:
    return "CLFFT_INVALID_KERNEL_DEFINITION";
  case CLFFT_INVALID_KERNEL_NAME:
    return "CLFFT_INVALID_KERNEL_NAME";
  case CLFFT_INVALID_PROGRAM_EXECUTABLE:
    return "CLFFT_INVALID_PROGRAM_EXECUTABLE";
  case CLFFT_INVALID_PROGRAM:
    return "CLFFT_INVALID_PROGRAM";
  case CLFFT_INVALID_BUILD_OPTIONS:
    return "CLFFT_INVALID_BUILD_OPTIONS";
  case CLFFT_INVALID_BINARY:
    return "CLFFT_INVALID_BINARY";
  case CLFFT_INVALID_SAMPLER:
    return "CLFFT_INVALID_SAMPLER";
  case CLFFT_INVALID_IMAGE_SIZE:
    return "CLFFT_INVALID_IMAGE_SIZE";
  case CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    return "CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR";
  case CLFFT_INVALID_MEM_OBJECT:
    return "CLFFT_INVALID_MEM_OBJECT";
  case CLFFT_INVALID_HOST_PTR:
    return "CLFFT_INVALID_HOST_PTR";
  case CLFFT_INVALID_COMMAND_QUEUE:
    return "CLFFT_INVALID_COMMAND_QUEUE";
  case CLFFT_INVALID_QUEUE_PROPERTIES:
    return "CLFFT_INVALID_QUEUE_PROPERTIES";
  case CLFFT_INVALID_CONTEXT:
    return "CLFFT_INVALID_CONTEXT";
  case CLFFT_INVALID_DEVICE:
    return "CLFFT_INVALID_DEVICE";
  case CLFFT_INVALID_PLATFORM:
    return "CLFFT_INVALID_PLATFORM";
  case CLFFT_INVALID_DEVICE_TYPE:
    return "CLFFT_INVALID_DEVICE_TYPE";
  case CLFFT_INVALID_VALUE:
    return "CLFFT_INVALID_VALUE";
  case CLFFT_MAP_FAILURE:
    return "CLFFT_MAP_FAILURE";
  case CLFFT_BUILD_PROGRAM_FAILURE:
    return "CLFFT_BUILD_PROGRAM_FAILURE";
  case CLFFT_IMAGE_FORMAT_NOT_SUPPORTED:
    return "CLFFT_IMAGE_FORMAT_NOT_SUPPORTED";
  case CLFFT_IMAGE_FORMAT_MISMATCH:
    return "CLFFT_IMAGE_FORMAT_MISMATCH";
  case CLFFT_MEM_COPY_OVERLAP:
    return "CLFFT_MEM_COPY_OVERLAP";
  case CLFFT_PROFILING_INFO_NOT_AVAILABLE:
    return "CLFFT_PROFILING_INFO_NOT_AVAILABLE";
  case CLFFT_OUT_OF_HOST_MEMORY:
    return "CLFFT_OUT_OF_HOST_MEMORY";
  case CLFFT_OUT_OF_RESOURCES:
    return "CLFFT_OUT_OF_RESOURCES";
  case CLFFT_MEM_OBJECT_ALLOCATION_FAILURE:
    return "CLFFT_MEM_OBJECT_ALLOCATION_FAILURE";
  case CLFFT_COMPILER_NOT_AVAILABLE:
    return "CLFFT_COMPILER_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_AVAILABLE:
    return "CLFFT_DEVICE_NOT_AVAILABLE";
  case CLFFT_DEVICE_NOT_FOUND:
    return "CLFFT_DEVICE_NOT_FOUND";
  case CLFFT_BUGCHECK:
    return "CLFFT_BUGCHECK";
  case CLFFT_NOTIMPLEMENTED:
    return "CLFFT_NOTIMPLEMENTED";
  case CLFFT_TRANSPOSED_NOTIMPLEMENTED:
    return "CLFFT_TRANSPOSED_NOTIMPLEMENTED";
  case CLFFT_FILE_NOT_FOUND:
    return "CLFFT_FILE_NOT_FOUND";
  case CLFFT_FILE_CREATE_FAILURE:
    return "CLFFT_FILE_CREATE_FAILURE";
  case CLFFT_VERSION_MISMATCH:
    return "CLFFT_VERSION_MISMATCH";
  case CLFFT_DEVICE_NO_DOUBLE:
    return "CLFFT_DEVICE_NO_DOUBLE";
  case CLFFT_DEVICE_MISMATCH:
    return "CLFFT_DEVICE_MISMATCH";
  default:
    return "CLFFT_UNKNOWN_ERROR";
  }
}
#endif  // USE FFT


#endif  // USE_GREENTEA



}  // namespace caffe
