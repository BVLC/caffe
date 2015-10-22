#ifndef CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_
#define CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include <boost/optional.hpp>
#include <boost/function.hpp>

namespace caffe {

bool deserialize(const char* data,
                 size_t size,
                 ::google::protobuf::Message& msg);

string serialize(const ::google::protobuf::Message& msg);


} //namespace caffe

#endif //CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_

