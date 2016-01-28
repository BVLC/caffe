#ifndef CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_
#define CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

bool deserialize(const char* data,
                 size_t size,
                 ::google::protobuf::Message* msg);

string serialize(const ::google::protobuf::Message& msg);


}  // namespace caffe

#endif  // CAFFE_SERIALIZATION_PROTOSERIALIZE_HPP_

