#ifndef CAFFE_UTIL_TYPE_UTILS_HPP_
#define CAFFE_UTIL_TYPE_UTILS_HPP_

#include <limits>

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

#define INVALID_DATA_INDEX -1
#define AUX_DATA_INDEX 0
#define HALF_DATA_INDEX 3
#define FLOAT_DATA_INDEX 1
#define DOUBLE_DATA_INDEX 2
#define UINT8_DATA_INDEX 4
#define UINT16_DATA_INDEX 5
#define UINT32_DATA_INDEX 6
#define UINT64_DATA_INDEX 7
#define INT8_DATA_INDEX 8
#define INT16_DATA_INDEX 9
#define INT32_DATA_INDEX 10
#define INT64_DATA_INDEX 11

#define PROTO_DATA_INDEX_MIN 1
#define PROTO_DATA_INDEX_MAX 11


template<typename T>
inline size_t safe_sizeof() {
  return sizeof(T);
}

template<>
inline size_t safe_sizeof<void>() {
  return 1;
}

template<>
inline size_t safe_sizeof<const void>() {
  return 1;
}

template<typename T>
inline string safe_type_name() {
  LOG(FATAL) << "Unknown type name" << std::endl;
  return "";
}

template<>
inline string safe_type_name<char>() {
  return "char";
}
template<>
inline string safe_type_name<bool>() {
  return "bool";
}
template<>
inline string safe_type_name<half_fp>() {
  return "half";
}
template<>
inline string safe_type_name<float>() {
  return "float";
}
template<>
inline string safe_type_name<double>() {
  return "double";
}
template<>
inline string safe_type_name<int8_t>() {
  return "int8_t";
}
template<>
inline string safe_type_name<int16_t>() {
  return "int16_t";
}
template<>
inline string safe_type_name<int32_t>() {
  return "int32_t";
}
template<>
inline string safe_type_name<int64_t>() {
  return "int64_t";
}
template<>
inline string safe_type_name<uint8_t>() {
  return "uint8_t";
}
template<>
inline string safe_type_name<uint16_t>() {
  return "uint16_t";
}
template<>
inline string safe_type_name<uint32_t>() {
  return "uint32_t";
}
template<>
inline string safe_type_name<uint64_t>() {
  return "uint64_t";
}
template<>
inline string safe_type_name<void>() {
  return "void";
}

template<typename T>
inline DataType proto_data_type() {
  LOG(FATAL) << "Unknown type" << std::endl;
  return CAFFE_HALF;  // Unreachable
}

template<>
inline DataType proto_data_type<half_fp>() {
  return CAFFE_HALF;
}
template<>
inline DataType proto_data_type<float>() {
  return CAFFE_FLOAT;
}
template<>
inline DataType proto_data_type<double>() {
  return CAFFE_DOUBLE;
}
template<>
inline DataType proto_data_type<int8_t>() {
  return CAFFE_INT8_QUANTIZED;
}
template<>
inline DataType proto_data_type<int16_t>() {
  return CAFFE_INT16_QUANTIZED;
}
template<>
inline DataType proto_data_type<int32_t>() {
  return CAFFE_INT32_QUANTIZED;
}
template<>
inline DataType proto_data_type<int64_t>() {
  return CAFFE_INT64_QUANTIZED;
}
template<>
inline DataType proto_data_type<uint8_t>() {
  return CAFFE_INT8_QUANTIZED;
}
template<>
inline DataType proto_data_type<uint16_t>() {
  return CAFFE_INT16_QUANTIZED;
}
template<>
inline DataType proto_data_type<uint32_t>() {
  return CAFFE_INT32_QUANTIZED;
}
template<>
inline DataType proto_data_type<uint64_t>() {
  return CAFFE_INT64_QUANTIZED;
}


template<typename T>
inline int_tp data_type_index() {
  LOG(FATAL) << "Unknown type" << std::endl;
  return INVALID_DATA_INDEX;  // Unreachable
}

template<>
inline int_tp data_type_index<half_fp>() {
  return HALF_DATA_INDEX;
}
template<>
inline int_tp data_type_index<float>() {
  return FLOAT_DATA_INDEX;
}
template<>
inline int_tp data_type_index<double>() {
  return DOUBLE_DATA_INDEX;
}
template<>
inline int_tp data_type_index<int8_t>() {
  return INT8_DATA_INDEX;
}
template<>
inline int_tp data_type_index<int16_t>() {
  return INT16_DATA_INDEX;
}
template<>
inline int_tp data_type_index<int32_t>() {
  return INT32_DATA_INDEX;
}
template<>
inline int_tp data_type_index<int64_t>() {
  return INT64_DATA_INDEX;
}
template<>
inline int_tp data_type_index<uint8_t>() {
  return UINT8_DATA_INDEX;
}
template<>
inline int_tp data_type_index<uint16_t>() {
  return UINT16_DATA_INDEX;
}
template<>
inline int_tp data_type_index<uint32_t>() {
  return UINT32_DATA_INDEX;
}
template<>
inline int_tp data_type_index<uint64_t>() {
  return UINT64_DATA_INDEX;
}

template<typename T>
inline double type_max_val() {
  LOG(FATAL) << "Unknown type" << std::endl;
  return 0.0;  // Unreachable
}

template<>
inline double type_max_val<half_fp>() {
  return static_cast<double>(HALF_MAX);
}
template<>
inline double type_max_val<float>() {
  return static_cast<double>(std::numeric_limits<float>::max());
}
template<>
inline double type_max_val<double>() {
  return static_cast<double>(std::numeric_limits<double>::max());
}
template<>
inline double type_max_val<int8_t>() {
  return static_cast<double>(std::numeric_limits<int8_t>::max());
}
template<>
inline double type_max_val<int16_t>() {
  return static_cast<double>(std::numeric_limits<int16_t>::max());
}
template<>
inline double type_max_val<int32_t>() {
  return static_cast<double>(std::numeric_limits<int32_t>::max());
}
template<>
inline double type_max_val<int64_t>() {
  return static_cast<double>(std::numeric_limits<int64_t>::max());
}
template<>
inline double type_max_val<uint8_t>() {
  return static_cast<double>(std::numeric_limits<uint8_t>::max());
}
template<>
inline double type_max_val<uint16_t>() {
  return static_cast<double>(std::numeric_limits<uint16_t>::max());
}
template<>
inline double type_max_val<uint32_t>() {
  return static_cast<double>(std::numeric_limits<uint32_t>::max());
}
template<>
inline double type_max_val<uint64_t>() {
  return static_cast<double>(std::numeric_limits<uint64_t>::max());
}

template<typename T>
inline double type_min_val() {
  LOG(FATAL) << "Unknown type" << std::endl;
  return 0.0;  // Unreachable
}

template<>
inline double type_min_val<half_fp>() {
  return static_cast<double>(-HALF_MAX);
}
template<>
inline double type_min_val<float>() {
  return -static_cast<double>(std::numeric_limits<float>::max());
}
template<>
inline double type_min_val<double>() {
  return -static_cast<double>(std::numeric_limits<double>::max());
}
template<>
inline double type_min_val<int8_t>() {
  return static_cast<double>(std::numeric_limits<int8_t>::min());
}
template<>
inline double type_min_val<int16_t>() {
  return static_cast<double>(std::numeric_limits<int16_t>::min());
}
template<>
inline double type_min_val<int32_t>() {
  return static_cast<double>(std::numeric_limits<int32_t>::min());
}
template<>
inline double type_min_val<int64_t>() {
  return static_cast<double>(std::numeric_limits<int64_t>::min());
}
template<>
inline double type_min_val<uint8_t>() {
  return static_cast<double>(std::numeric_limits<uint8_t>::min());
}
template<>
inline double type_min_val<uint16_t>() {
  return static_cast<double>(std::numeric_limits<uint16_t>::min());
}
template<>
inline double type_min_val<uint32_t>() {
  return static_cast<double>(std::numeric_limits<uint32_t>::min());
}
template<>
inline double type_min_val<uint64_t>() {
  return static_cast<double>(std::numeric_limits<uint64_t>::min());
}

template<typename T>
inline size_t type_max_integer_representable() {
  LOG(FATAL) << "Unknown type" << std::endl;
  return 0;
}

template<>
inline size_t type_max_integer_representable<half_fp>() {
  return 2048ULL;  // Floating point 2^11
}

template<>
inline size_t type_max_integer_representable<float>() {
  return 16777216ULL;  // Floating point 2^24
}

template<>
inline size_t type_max_integer_representable<double>() {
  return 9007199254740992ULL;  // Floating point 2^53
}

template<>
inline size_t type_max_integer_representable<int8_t>() {
  return std::numeric_limits<int8_t>::max();
}

template<>
inline size_t type_max_integer_representable<int16_t>() {
  return std::numeric_limits<int16_t>::max();
}

template<>
inline size_t type_max_integer_representable<int32_t>() {
  return std::numeric_limits<int32_t>::max();
}

template<>
inline size_t type_max_integer_representable<int64_t>() {
  return std::numeric_limits<int64_t>::max();
}

template<>
inline size_t type_max_integer_representable<uint8_t>() {
  return std::numeric_limits<uint8_t>::max();
}

template<>
inline size_t type_max_integer_representable<uint16_t>() {
  return std::numeric_limits<uint16_t>::max();
}

template<>
inline size_t type_max_integer_representable<uint32_t>() {
  return std::numeric_limits<uint32_t>::max();
}

template<>
inline size_t type_max_integer_representable<uint64_t>() {
  return std::numeric_limits<uint64_t>::max();
}


template<typename T>
inline T type_round(T x) {
  return x;
}

template<>
inline float type_round(float x) {
  return std::round(x);
}

template<>
inline double type_round(double x) {
  return std::round(x);
}

template<>
inline half_fp type_round(half_fp x) {
  return std::round(x);
}



}  // namespace caffe

#endif  // CAFFE_UTIL_TYPE_UTILS_HPP_
