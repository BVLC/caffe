#ifndef CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
#define CAFFE_BACKEND_DEVICE_PROGRAM_HPP_

#include <string>
#include "caffe/backend/backend.hpp"
#include "device_kernel.hpp"

namespace caffe {

class Device;

class DeviceProgram {

 public:
  virtual void Compile(bool load_cache, bool store_cache) = 0;
  virtual shared_ptr<DeviceKernel> GetKernel(string name) = 0;

  virtual string function(string name,
          vector<KernelArg> args) = 0;
  virtual string kernel_loop(string type, string index, string n) = 0;
  virtual string setup() = 0;
  virtual string atomics() = 0;
  virtual string global_ptr(string type, string name) = 0;
  virtual string local_ptr(string type, string name) = 0;
  virtual string local_mem(string type) = 0;

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

  void set_compile_flags(uint32_t flags);
  void set_source(string src);
  void add_source(string src);

  string string_identifier();

  template<typename Dtype>
  string atomic_add(string source, string operand);

  template<typename Dtype>
  string define_type(string name);

  template<typename Dtype>
  KernelArg create_kernel_arg(string name, uint64_t flags);

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

  uint64_t compile_flags_;
  Device *device_;
  string src_;
  bool src_has_changed_;
  string string_identifier_;
  std::map<string, KernelArgs> args_;

};

}

#endif  // CAFFE_BACKEND_DEVICE_PROGRAM_HPP_
