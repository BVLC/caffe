#ifndef CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_DEVICE_PROGRAM_HPP_

#include <string>
#include "caffe/backend/backend.hpp"
#include "device_kernel.hpp"

namespace caffe {

class Device;

class DeviceProgram {

 public:
  inline const Device* device() {
    return device_;
  }

  virtual bool Compile(bool load_cache, bool store_cache) = 0;
  virtual shared_ptr<DeviceKernel> GetKernel(string name) = 0;

  virtual string function(string name, KernelArgs args,
                          KernelHints hints = KernelHints()) = 0;
  virtual string kernel_loop(string type, string index, string n) = 0;
  virtual string setup() = 0;
  virtual string atomics() = 0;
  virtual string vector_accessors();
  virtual string global_ptr(string type, string name) = 0;
  virtual string local_ptr(string type, string name) = 0;
  virtual string local_mem(string type, string name) = 0;

  virtual string local_id(uint_tp fixed_index) = 0;
  virtual string local_id(string runtime_index) = 0;
  virtual string local_size(uint_tp fixed_index) = 0;
  virtual string local_size(string runtime_index) = 0;
  virtual string group_id(uint_tp fixed_index) = 0;
  virtual string group_id(string runtime_index) = 0;
  virtual string group_size(uint_tp fixed_index) = 0;
  virtual string group_size(string runtime_index) = 0;
  virtual string global_id(uint_tp fixed_index) = 0;
  virtual string global_id(string runtime_index) = 0;
  virtual string global_size(uint_tp fixed_index) = 0;
  virtual string global_size(string runtime_index) = 0;

  virtual string local_barrier() = 0;
  virtual string global_barrier() = 0;

  void set_compile_flags(uint64_t flags);
  void set_source(string src);
  void add_source(string src);

  int64_t identifier();
  string string_identifier();

  template<typename Dtype>
  string device_type_name() const;

  virtual string device_type_name_void() const = 0;
  virtual string device_type_name_bool() const = 0;
  virtual string device_type_name_char() const = 0;
  virtual string device_type_name_half() const = 0;
  virtual string device_type_name_float() const = 0;
  virtual string device_type_name_double() const = 0;
  virtual string device_type_name_int8() const = 0;
  virtual string device_type_name_int16() const = 0;
  virtual string device_type_name_int32() const = 0;
  virtual string device_type_name_int64() const = 0;
  virtual string device_type_name_uint8() const = 0;
  virtual string device_type_name_uint16() const = 0;
  virtual string device_type_name_uint32() const = 0;
  virtual string device_type_name_uint64() const = 0;

  template<typename Dtype>
  string convert_type(int_tp vec_len, string src_val) const;

  virtual string convert_type_char(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_half(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_float(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_double(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_uint8(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_uint16(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_uint32(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_uint64(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_int8(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_int16(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_int32(int_tp vec_len, string src_val) const = 0;
  virtual string convert_type_int64(int_tp vec_len, string src_val) const = 0;

  template<typename Dtype>
  string atomic_add(string source, string operand);

  template<typename Dtype>
  string helper_functions();

  template<typename Dtype>
  string define_type(const char* name);

  template<typename Dtype>
  string define_type(string name);

  template<typename Dtype>
  string define(const char* name, Dtype value);

  template<typename Dtype>
  string define(string name, Dtype value);

  template<typename Dtype>
  string define_vector_type(const char* name, int_tp from, int_tp to);

  template<typename Dtype>
  string define_vector_type(string name, int_tp from, int_tp to);

  template<typename Dtype>
  KernelArg create_kernel_arg(string name, uint64_t flags = KERNEL_ARG_NONE);

  KernelHint create_kernel_hint(KernelHintOption option, string value);
  KernelHint create_kernel_hint(KernelHintOption option, int8_t value);
  KernelHint create_kernel_hint(KernelHintOption option, int16_t value);
  KernelHint create_kernel_hint(KernelHintOption option, int32_t value);
  KernelHint create_kernel_hint(KernelHintOption option, int64_t value);

 protected:
  DeviceProgram(Device* dev);

  virtual string kernel_arg_type_void(uint64_t flags) = 0;
  virtual string kernel_arg_type_bool(uint64_t flags) = 0;
  virtual string kernel_arg_type_char(uint64_t flags) = 0;
  virtual string kernel_arg_type_half(uint64_t flags) = 0;
  virtual string kernel_arg_type_float(uint64_t flags) = 0;
  virtual string kernel_arg_type_double(uint64_t flags) = 0;
  virtual string kernel_arg_type_int8_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_int16_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_int32_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_int64_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_uint8_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_uint16_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_uint32_t(uint64_t flags) = 0;
  virtual string kernel_arg_type_uint64_t(uint64_t flags) = 0;

  virtual string helper_functions_half() const = 0;
  virtual string helper_functions_float() const = 0;
  virtual string helper_functions_double() const = 0;
  virtual string helper_functions_uint8() const = 0;
  virtual string helper_functions_uint16() const = 0;
  virtual string helper_functions_uint32() const = 0;
  virtual string helper_functions_uint64() const = 0;


  uint64_t compile_flags_;
  Device *device_;
  string src_;
  bool src_has_changed_;
  int64_t identifier_;
  std::map<string, KernelArgs> args_;
};

}

#endif  // CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
