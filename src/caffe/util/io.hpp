// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <string>
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

void ReadImageToProto(const string& filename, BlobProto* proto);
void WriteProtoToImage(const string& filename, const BlobProto& proto);


}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
