// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <google/protobuf/message.h>

#include <string>

#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

void ReadImageToProto(const string& filename, BlobProto* proto);
void WriteProtoToImage(const string& filename, const BlobProto& proto);

void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto);
inline void ReadProtoFromTextFile(const string& filename,
    ::google::protobuf::Message* proto) {
  ReadProtoFromTextFile(filename.c_str(), proto);
}

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
