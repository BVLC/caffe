#ifndef CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_

#include "caffe/backend/device_program.hpp"
#include "caffe/backend/cuda/cuda_device_kernel.hpp"

#ifdef USE_CUDA
#include "cuda.h"
#include "nvrtc.h"
#endif  // USE_CUDA

namespace caffe {

#ifdef USE_CUDA

class CudaDeviceProgram : public DeviceProgram {
 public:
  explicit CudaDeviceProgram(Device *dev);

  virtual void Compile(bool load_cache, bool store_cache);
  virtual shared_ptr<DeviceKernel> GetKernel(string name);

  virtual string function(string name, KernelArgs args);
  virtual string kernel_loop(string type, string index, string n);
  virtual string setup();
  virtual string atomics();
  virtual string global_ptr(string type, string name);
  virtual string local_ptr(string type, string name);
  virtual string local_mem(string type, string name);

  virtual string local_id(uint_tp fixed_index);
  virtual string local_id(string runtime_index);
  virtual string local_size(uint_tp fixed_index);
  virtual string local_size(string runtime_index);
  virtual string group_id(uint_tp fixed_index);
  virtual string group_id(string runtime_index);
  virtual string group_size(uint_tp fixed_index);
  virtual string group_size(string runtime_index);
  virtual string global_id(uint_tp fixed_index);
  virtual string global_id(string runtime_index);
  virtual string global_size(uint_tp fixed_index);
  virtual string global_size(string runtime_index);

  virtual string local_barrier();
  virtual string global_barrier();

  virtual string kernel_arg_type_void(uint64_t flags);
  virtual string kernel_arg_type_bool(uint64_t flags);
  virtual string kernel_arg_type_char(uint64_t flags);
  virtual string kernel_arg_type_half(uint64_t flags);
  virtual string kernel_arg_type_float(uint64_t flags);
  virtual string kernel_arg_type_double(uint64_t flags);
  virtual string kernel_arg_type_int8_t(uint64_t flags);
  virtual string kernel_arg_type_int16_t(uint64_t flags);
  virtual string kernel_arg_type_int32_t(uint64_t flags);
  virtual string kernel_arg_type_int64_t(uint64_t flags);
  virtual string kernel_arg_type_uint8_t(uint64_t flags);
  virtual string kernel_arg_type_uint16_t(uint64_t flags);
  virtual string kernel_arg_type_uint32_t(uint64_t flags);
  virtual string kernel_arg_type_uint64_t(uint64_t flags);

 private:
  nvrtcProgram cuda_program_;
  CUmodule cuda_module_;

};

#endif  // USE_CUDA

}

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_
