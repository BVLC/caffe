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
  ~CudaDeviceProgram();

  virtual bool Compile(bool load_cache, bool store_cache);
  virtual shared_ptr<DeviceKernel> GetKernel(string name);

  virtual string function(string name, KernelArgs args,
                          KernelHints hints = KernelHints());
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

  virtual string device_type_name_void() const;
  virtual string device_type_name_bool() const;
  virtual string device_type_name_char() const;
  virtual string device_type_name_half() const;
  virtual string device_type_name_float() const;
  virtual string device_type_name_double() const;
  virtual string device_type_name_int8() const;
  virtual string device_type_name_int16() const;
  virtual string device_type_name_int32() const;
  virtual string device_type_name_int64() const;
  virtual string device_type_name_uint8() const;
  virtual string device_type_name_uint16() const;
  virtual string device_type_name_uint32() const;
  virtual string device_type_name_uint64() const;

  virtual string convert_type_char(int_tp vec_len, string src_val) const;
  virtual string convert_type_half(int_tp vec_len, string src_val) const;
  virtual string convert_type_float(int_tp vec_len, string src_val) const;
  virtual string convert_type_double(int_tp vec_len, string src_val) const;
  virtual string convert_type_uint8(int_tp vec_len, string src_val) const;
  virtual string convert_type_uint16(int_tp vec_len, string src_val) const;
  virtual string convert_type_uint32(int_tp vec_len, string src_val) const;
  virtual string convert_type_uint64(int_tp vec_len, string src_val) const;
  virtual string convert_type_int8(int_tp vec_len, string src_val) const;
  virtual string convert_type_int16(int_tp vec_len, string src_val) const;
  virtual string convert_type_int32(int_tp vec_len, string src_val) const;
  virtual string convert_type_int64(int_tp vec_len, string src_val) const;

  virtual string helper_functions_half() const;
  virtual string helper_functions_float() const;
  virtual string helper_functions_double() const;
  virtual string helper_functions_uint8() const;
  virtual string helper_functions_uint16() const;
  virtual string helper_functions_uint32() const;
  virtual string helper_functions_uint64() const;

 private:
  shared_ptr<nvrtcProgram> cuda_program_;
  shared_ptr<CUmodule> cuda_module_;

};

#endif  // USE_CUDA

}

#endif  // CAFFE_BACKEND_CUDA_CUDA_DEVICE_PROGRAM_HPP_
