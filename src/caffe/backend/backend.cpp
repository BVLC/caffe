#include "caffe/backend/backend.hpp"

namespace caffe {

template<>
size_t safe_sizeof<void>() {
  return 1;
}

template<>
size_t safe_sizeof<const void>() {
  return 1;
}

template<typename T>
size_t safe_sizeof() {
  return sizeof(T);
}

template<> size_t safe_sizeof<char>();
template<> size_t safe_sizeof<bool>();
template<> size_t safe_sizeof<half_float::half>();
template<> size_t safe_sizeof<float>();
template<> size_t safe_sizeof<double>();
template<> size_t safe_sizeof<int8_t>();
template<> size_t safe_sizeof<int16_t>();
template<> size_t safe_sizeof<int32_t>();
template<> size_t safe_sizeof<int64_t>();
template<> size_t safe_sizeof<uint8_t>();
template<> size_t safe_sizeof<uint16_t>();
template<> size_t safe_sizeof<uint32_t>();
template<> size_t safe_sizeof<uint64_t>();

template<> size_t safe_sizeof<const char>();
template<> size_t safe_sizeof<const bool>();
template<> size_t safe_sizeof<const half_float::half>();
template<> size_t safe_sizeof<const float>();
template<> size_t safe_sizeof<const double>();
template<> size_t safe_sizeof<const int8_t>();
template<> size_t safe_sizeof<const int16_t>();
template<> size_t safe_sizeof<const int32_t>();
template<> size_t safe_sizeof<const int64_t>();
template<> size_t safe_sizeof<const uint8_t>();
template<> size_t safe_sizeof<const uint16_t>();
template<> size_t safe_sizeof<const uint32_t>();
template<> size_t safe_sizeof<const uint64_t>();


template<>
string safe_type_name<char>() {
  return "char";
}
template<>
string safe_type_name<bool>() {
  return "bool";
}
template<>
string safe_type_name<half_float::half>() {
  return "half";
}
template<>
string safe_type_name<float>() {
  return "float";
}
template<>
string safe_type_name<double>() {
  return "double";
}
template<>
string safe_type_name<int8_t>() {
  return "int8_t";
}
template<>
string safe_type_name<int16_t>() {
  return "int16_t";
}
template<>
string safe_type_name<int32_t>() {
  return "int32_t";
}
template<>
string safe_type_name<int64_t>() {
  return "int64_t";
}
template<>
string safe_type_name<uint8_t>() {
  return "uint8_t";
}
template<>
string safe_type_name<uint16_t>() {
  return "uint16_t";
}
template<>
string safe_type_name<uint32_t>() {
  return "uint32_t";
}
template<>
string safe_type_name<uint64_t>() {
  return "uint64_t";
}
template<>
string safe_type_name<void>() {
  return "void";
}

}  // namespace caffe
