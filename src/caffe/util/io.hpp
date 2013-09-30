// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <google/protobuf/message.h>

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using ::google::protobuf::Message;

namespace caffe {

void ReadImageToProto(const string& filename, BlobProto* proto);

template <typename Dtype>
inline void ReadImageToBlob(const string& filename, Blob<Dtype>* blob) {
  BlobProto proto;
  ReadImageToProto(filename, &proto);
  blob->FromProto(proto);
}

void WriteProtoToImage(const string& filename, const BlobProto& proto);

template <typename Dtype>
inline void WriteBlobToImage(const string& filename, const Blob<Dtype>& blob) {
  BlobProto proto;
  blob.ToProto(&proto);
  WriteProtoToImage(filename, proto);
}

void ReadProtoFromTextFile(const char* filename,
    Message* proto);
inline void ReadProtoFromTextFile(const string& filename,
    Message* proto) {
  ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

void ReadProtoFromBinaryFile(const char* filename,
    Message* proto);
inline void ReadProtoFromBinaryFile(const string& filename,
    Message* proto) {
  ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}


}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
