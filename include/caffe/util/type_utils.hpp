#ifndef CAFFE_UTIL_TYPE_UTILS_HPP_
#define CAFFE_UTIL_TYPE_UTILS_HPP_

#include "caffe/definitions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename T>
size_t safe_sizeof();

template<typename T>
string safe_type_name();

template<typename T>
DataType proto_data_type();

}


#endif  // CAFFE_UTIL_TYPE_UTILS_HPP_
