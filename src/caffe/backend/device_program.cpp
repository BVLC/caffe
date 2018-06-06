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
  src_has_changed_ = true;
}

void DeviceProgram::add_source(string src) {
  this->src_ += src;
  src_has_changed_ = true;
}

void DeviceProgram::set_compile_flags(uint64_t flags) {
  this->compile_flags_ = flags;
}

template<typename Dtype>
string DeviceProgram::atomic_add(string source, string operand) {
  return "caffe_gpu_atomic_" + safe_type_name<Dtype>()
          + "_add(" + source + ", " + operand + ");";
}

string DeviceProgram::vector_accessors() {
  stringstream ss;

  vector<string> elems4({
      "x", "y", "z", "w" });
  vector<string> elems16({
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      "s8", "s9", "sA", "sB", "sC", "sD", "sE", "sF" });

  for (int_tp i = 1; i <= 16; i *= 2) {
    for (int_tp j = 0; j < i; ++j) {
      if (i == 1) {
        ss << "#define VEC_" << i << "_" << j << "(ELEM)"
           << " ELEM" << std::endl;
      } else if (i < 8) {
        ss << "#define VEC_" << i << "_" << j << "(ELEM)"
           << " ELEM." << elems4[j] << std::endl;
      } else {
        ss << "#define VEC_" << i << "_" << j << "(ELEM)"
           << " ELEM." << elems16[j] << std::endl;
      }
    }
  }
  return ss.str();
}

template
string DeviceProgram::atomic_add<half_fp>(string source, string operand);
template
string DeviceProgram::atomic_add<float>(string source, string operand);
template
string DeviceProgram::atomic_add<double>(string source, string operand);
template
string DeviceProgram::atomic_add<uint8_t>(string source, string operand);
template
string DeviceProgram::atomic_add<uint16_t>(string source, string operand);
template
string DeviceProgram::atomic_add<uint32_t>(string source, string operand);
template
string DeviceProgram::atomic_add<uint64_t>(string source, string operand);

template<typename Dtype>
string DeviceProgram::define_type(const char* name) {
  stringstream ss;
  ss << "#ifdef " << name << std::endl;
  ss << "#undef " << name << std::endl;
  ss << "#endif  //" << name << std::endl;
  ss << "#define " << name << " " << safe_type_name<Dtype>() << std::endl;
  return ss.str();
}

template string DeviceProgram::define_type<bool>(const char* name);
template string DeviceProgram::define_type<char>(const char* name);
template string DeviceProgram::define_type<half_fp>(const char* name);
template string DeviceProgram::define_type<float>(const char* name);
template string DeviceProgram::define_type<double>(const char* name);
template string DeviceProgram::define_type<int8_t>(const char* name);
template string DeviceProgram::define_type<int16_t>(const char* name);
template string DeviceProgram::define_type<int32_t>(const char* name);
template string DeviceProgram::define_type<int64_t>(const char* name);
template string DeviceProgram::define_type<uint8_t>(const char* name);
template string DeviceProgram::define_type<uint16_t>(const char* name);
template string DeviceProgram::define_type<uint32_t>(const char* name);
template string DeviceProgram::define_type<uint64_t>(const char* name);

template<typename Dtype>
string DeviceProgram::define_type(string name) {
  stringstream ss;
  ss << "#ifdef " << name << std::endl;
  ss << "#undef " << name << std::endl;
  ss << "#endif  //" << name << std::endl;
  ss << "#define " << name << " " << safe_type_name<Dtype>() << std::endl;
  return ss.str();
}

template string DeviceProgram::define_type<bool>(string name);
template string DeviceProgram::define_type<char>(string name);
template string DeviceProgram::define_type<half_fp>(string name);
template string DeviceProgram::define_type<float>(string name);
template string DeviceProgram::define_type<double>(string name);
template string DeviceProgram::define_type<int8_t>(string name);
template string DeviceProgram::define_type<int16_t>(string name);
template string DeviceProgram::define_type<int32_t>(string name);
template string DeviceProgram::define_type<int64_t>(string name);
template string DeviceProgram::define_type<uint8_t>(string name);
template string DeviceProgram::define_type<uint16_t>(string name);
template string DeviceProgram::define_type<uint32_t>(string name);
template string DeviceProgram::define_type<uint64_t>(string name);

template<typename Dtype>
string DeviceProgram::define(const char* name, Dtype value) {
  stringstream ss;
  stringstream lhss;
  lhss << name;
  stringstream rhss;
  rhss << value;
  if (!(lhss.str() == rhss.str())) {
    ss << "#ifdef " << name << std::endl;
    ss << "#undef " << name << std::endl;
    ss << "#endif  //" << name << std::endl;
    ss << "#define " << name << " " << value << std::endl;
  }
  return ss.str();
}

template string DeviceProgram::define<bool>(const char* name, bool value);
template string DeviceProgram::define<char>(const char* name, char value);
template string DeviceProgram::define<half_fp>(const char* name, half_fp value);
template string DeviceProgram::define<float>(const char* name, float value);
template string DeviceProgram::define<double>(const char* name, double value);
template string DeviceProgram::define<int8_t>(const char* name, int8_t value);
template string DeviceProgram::define<int16_t>(const char* name, int16_t value);
template string DeviceProgram::define<int32_t>(const char* name, int32_t value);
template string DeviceProgram::define<int64_t>(const char* name, int64_t value);
template string DeviceProgram::define<uint8_t>(const char* name, uint8_t value);
template string DeviceProgram::define<uint16_t>(const char* name,
                                                uint16_t value);
template string DeviceProgram::define<uint32_t>(const char* name,
                                                uint32_t value);
template string DeviceProgram::define<uint64_t>(const char* name,
                                                uint64_t value);
template string DeviceProgram::define<string>(const char* name, string value);
template string DeviceProgram::define<const char*>(const char* name,
                                                   const char* value);

template<typename Dtype>
string DeviceProgram::define(string name, Dtype value) {
  stringstream ss;
  stringstream lhss;
  lhss << name;
  stringstream rhss;
  rhss << value;
  if (!(lhss.str() == rhss.str())) {
    ss << "#ifdef " << name << std::endl;
    ss << "#undef " << name << std::endl;
    ss << "#endif  //" << name << std::endl;
    ss << "#define " << name << " " << value << std::endl;
  }
  return ss.str();
}

template string DeviceProgram::define<bool>(string name, bool value);
template string DeviceProgram::define<char>(string name, char value);
template string DeviceProgram::define<half_fp>(string name, half_fp value);
template string DeviceProgram::define<float>(string name, float value);
template string DeviceProgram::define<double>(string name, double value);
template string DeviceProgram::define<int8_t>(string name, int8_t value);
template string DeviceProgram::define<int16_t>(string name, int16_t value);
template string DeviceProgram::define<int32_t>(string name, int32_t value);
template string DeviceProgram::define<int64_t>(string name, int64_t value);
template string DeviceProgram::define<uint8_t>(string name, uint8_t value);
template string DeviceProgram::define<uint16_t>(string name, uint16_t value);
template string DeviceProgram::define<uint32_t>(string name, uint32_t value);
template string DeviceProgram::define<uint64_t>(string name, uint64_t value);
template string DeviceProgram::define<string>(string name, string value);
template string DeviceProgram::define<const char*>(string name,
                                                   const char* value);

template<typename Dtype>
string DeviceProgram::define_vector_type(const char* name, int_tp from,
                                         int_tp to) {
  return define_vector_type<Dtype>(string(name), from, to);
}

template string DeviceProgram::define_vector_type<bool>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<char>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<half_fp>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<float>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<double>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int8_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int16_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int32_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int64_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint8_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint16_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint32_t>(const char* name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint64_t>(const char* name,
                                                        int_tp from, int_tp to);

template<typename Dtype>
string DeviceProgram::define_vector_type(string name, int_tp from,
                                         int_tp to) {
  stringstream ss;
  int_tp safe_from = 2;
  if (from <= 1) {
    for (int_tp i = from; i <= 1; ++i) {
      ss << this->define_type<Dtype>(name);
      ss << this->define_type<Dtype>(name + std::to_string(i));
    }
  }
  safe_from = std::max(from, safe_from);
  for (int_tp i = safe_from; i <= to; i *= 2) {
    ss << this->define(name + std::to_string(i),
                       this->device_type_name<Dtype>() + std::to_string(i));
  }
  return ss.str();
}

template string DeviceProgram::define_vector_type<bool>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<char>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<half_fp>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<float>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<double>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int8_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int16_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int32_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<int64_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint8_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint16_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint32_t>(string name,
                                                        int_tp from, int_tp to);
template string DeviceProgram::define_vector_type<uint64_t>(string name,
                                                        int_tp from, int_tp to);

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

KernelHint DeviceProgram::create_kernel_hint(KernelHintOption option,
                                             string value) {
  return std::make_tuple(option, value);
}
KernelHint DeviceProgram::create_kernel_hint(KernelHintOption option,
                                             int8_t value) {
  return std::make_tuple(option, std::to_string(value));
}
KernelHint DeviceProgram::create_kernel_hint(KernelHintOption option,
                                             int16_t value) {
  return std::make_tuple(option, std::to_string(value));
}
KernelHint DeviceProgram::create_kernel_hint(KernelHintOption option,
                                             int32_t value) {
  return std::make_tuple(option, std::to_string(value));
}
KernelHint DeviceProgram::create_kernel_hint(KernelHintOption option,
                                             int64_t value) {
  return std::make_tuple(option, std::to_string(value));
}

string DeviceProgram::string_identifier() {
  return hash_hex_string(identifier_);
}

int64_t DeviceProgram::identifier() {
  if (src_has_changed_) {
    vector<string> factors;
    factors.push_back(this->device_->name());
    factors.push_back(src_);
    identifier_ = generate_hash(factors);
  }
  return identifier_;
}

template<>
string DeviceProgram::device_type_name<void>() const {
  return device_type_name_void();
}
template<>
string DeviceProgram::device_type_name<bool>() const {
  return device_type_name_bool();
}
template<>
string DeviceProgram::device_type_name<char>() const {
  return device_type_name_char();
}
template<>
string DeviceProgram::device_type_name<half_fp>() const {
  return device_type_name_half();
}
template<>
string DeviceProgram::device_type_name<float>() const {
  return device_type_name_float();
}
template<>
string DeviceProgram::device_type_name<double>() const {
  return device_type_name_double();
}
template<>
string DeviceProgram::device_type_name<int8_t>() const {
  return device_type_name_int8();
}
template<>
string DeviceProgram::device_type_name<int16_t>() const {
  return device_type_name_int16();
}
template<>
string DeviceProgram::device_type_name<int32_t>() const {
  return device_type_name_int32();
}
template<>
string DeviceProgram::device_type_name<int64_t>() const {
  return device_type_name_int64();
}
template<>
string DeviceProgram::device_type_name<uint8_t>() const {
  return device_type_name_uint8();
}
template<>
string DeviceProgram::device_type_name<uint16_t>() const {
  return device_type_name_uint16();
}
template<>
string DeviceProgram::device_type_name<uint32_t>() const {
  return device_type_name_uint32();
}
template<>
string DeviceProgram::device_type_name<uint64_t>() const {
  return device_type_name_uint64();
}


template<>
string DeviceProgram::convert_type<char>(int_tp vec_len,
                                         string src_val) const {
  return convert_type_char(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<half_fp>(int_tp vec_len,
                                            string src_val) const {
  return convert_type_half(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<float>(int_tp vec_len,
                                          string src_val) const {
  return convert_type_float(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<double>(int_tp vec_len,
                                           string src_val) const {
  return convert_type_double(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<uint8_t>(int_tp vec_len,
                                            string src_val) const {
  return convert_type_uint8(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<uint16_t>(int_tp vec_len,
                                             string src_val) const {
  return convert_type_uint16(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<uint32_t>(int_tp vec_len,
                                             string src_val) const {
  return convert_type_uint32(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<uint64_t>(int_tp vec_len,
                                             string src_val) const {
  return convert_type_uint64(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<int8_t>(int_tp vec_len,
                                           string src_val) const {
  return convert_type_int8(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<int16_t>(int_tp vec_len,
                                            string src_val) const {
  return convert_type_int16(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<int32_t>(int_tp vec_len,
                                            string src_val)const {
  return convert_type_int32(vec_len, src_val);
}
template<>
string DeviceProgram::convert_type<int64_t>(int_tp vec_len,
                                            string src_val) const {
  return convert_type_int64(vec_len, src_val);
}


template<>
string DeviceProgram::helper_functions<half_fp>() {
  return helper_functions_half();
}
template<>
string DeviceProgram::helper_functions<float>() {
  return helper_functions_float();
}
template<>
string DeviceProgram::helper_functions<double>() {
  return helper_functions_double();
}
template<>
string DeviceProgram::helper_functions<uint8_t>() {
  return helper_functions_uint8();
}
template<>
string DeviceProgram::helper_functions<uint16_t>() {
  return helper_functions_uint16();
}
template<>
string DeviceProgram::helper_functions<uint32_t>() {
  return helper_functions_uint32();
}
template<>
string DeviceProgram::helper_functions<uint64_t>() {
  return helper_functions_uint64();
}


}  // namespace caffe
