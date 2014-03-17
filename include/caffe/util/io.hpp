// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"
#include "caffe/proto/caffe.pb.h"

#include "boost/scoped_ptr.hpp"
#include "caffe/blob.hpp"

using std::string;
using ::google::protobuf::Message;

namespace caffe {

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
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

template <typename Dtype>
void hdf5_load_nd_dataset(
  hid_t file_id, const char* dataset_name_,
  int min_dim,//inclusive
  int max_dim,//inclusive
  //output:
  boost::scoped_ptr<Dtype>* array,
  std::vector<hsize_t>& dims
  );

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
