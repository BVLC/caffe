#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <string>
#include "caffe/serialization/ProtoSerialize.hpp"

namespace caffe {

bool deserialize(const char* data,
                 size_t size,
                 ::google::protobuf::Message* msg) {
  static const size_t max_decode_size = 300 * 1024 * 1024;
  using google::protobuf::io::ArrayInputStream;
  using google::protobuf::io::CodedInputStream;
  ArrayInputStream zero_stream(data, size);
  CodedInputStream coded_stream(&zero_stream);
  coded_stream.SetTotalBytesLimit(max_decode_size, max_decode_size);
  bool ret = msg->ParseFromCodedStream(&coded_stream);
  if (!ret) {
    LOG(ERROR) << "Parsing BlobUpdate failed";
  }
  return ret;
}

string serialize(const ::google::protobuf::Message& msg) {
  string str;
  msg.SerializeToString(&str);
  return str;
}

}  // namespace caffe

