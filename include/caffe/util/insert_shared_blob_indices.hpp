#ifndef CAFFE_UTIL_INSERT_SHARED_BLOB_INDICES_HPP_
#define CAFFE_UTIL_INSERT_SHARED_BLOB_INDICES_HPP_


#include "caffe/proto/caffe.pb.h"

namespace caffe {

int_tp InsertSharedBlobIndices(const NetParameter& param,
                               NetParameter* shared_memory_net_param);

}  // namespace caffe


#endif  // CAFFE_UTIL_INSERT_SHARED_BLOB_INDICES_HPP_
