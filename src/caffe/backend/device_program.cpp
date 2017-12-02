#include "caffe/backend/device_program.hpp"
#include "caffe/util/half_fp.hpp"
#include "caffe/util/hash.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/util/type_utils.hpp"


namespace caffe {

DeviceProgram::DeviceProgram(Device* dev) :
    device_(dev), src_(""), src_has_changed_(true) {

}

void DeviceProgram::set_source(string src) {
  this->src_ = src;
}

void DeviceProgram::add_source(string src) {
  this->src_ += src;
}

void DeviceProgram::set_compile_flags(uint32_t flags) {
  this->compile_flags_ = flags;
}

template<typename Dtype>
string DeviceProgram::atomic_add(string source, string operand) {
  return "caffe_gpu_atomic_" + safe_type_name<Dtype>()
          + "_add(" + source + ", " + operand + ")";
}

template
string DeviceProgram::atomic_add<half_fp>(string source,
                                                   string operand);
template
string DeviceProgram::atomic_add<float>(string source, string operand);
template
string DeviceProgram::atomic_add<double>(string source, string operand);

template<typename Dtype>
string DeviceProgram::define_type(string name) {
  stringstream ss;
  ss << "#ifdef " << name << std::endl;
  ss << "#undef " << name << std::endl;
  ss << "#endif  //" << name << std::endl;
  ss << "#define " << name << " " << safe_type_name<Dtype>() << std::endl;
  return ss.str();
}

template
string DeviceProgram::define_type<bool>(string name);
template
string DeviceProgram::define_type<char>(string name);
template
string DeviceProgram::define_type<half_fp>(string name);
template
string DeviceProgram::define_type<float>(string name);
template
string DeviceProgram::define_type<double>(string name);
template
string DeviceProgram::define_type<int8_t>(string name);
template
string DeviceProgram::define_type<int16_t>(string name);
template
string DeviceProgram::define_type<int32_t>(string name);
template
string DeviceProgram::define_type<int64_t>(string name);
template
string DeviceProgram::define_type<uint8_t>(string name);
template
string DeviceProgram::define_type<uint16_t>(string name);
template
string DeviceProgram::define_type<uint32_t>(string name);
template
string DeviceProgram::define_type<uint64_t>(string name);

template<>
KernelArg DeviceProgram::create_kernel_arg<void>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_void(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<bool>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_bool(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<char>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_char(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<half_fp>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_half(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<float>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_float(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<double>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_double(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<int8_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_int8_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<int16_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_int16_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<int32_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_int32_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<int64_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_int64_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<uint8_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_uint8_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<uint16_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_uint16_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<uint32_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_uint32_t(flags);
  return std::make_tuple(type, name, flags);
}

template<>
KernelArg DeviceProgram::create_kernel_arg<uint64_t>(
    string name, uint64_t flags) {
  string type = this->kernel_arg_type_uint64_t(flags);
  return std::make_tuple(type, name, flags);
}

string DeviceProgram::string_identifier() {
  if (src_has_changed_) {
    vector<string> factors;
    factors.push_back(this->device_->name());
    factors.push_back(src_);
    string_identifier_ = generate_hash(factors);
  }
  return string_identifier_;
}

}  // namespace caffe
