#ifndef CAFFE_SERIALIZATION_BLOBCODEC_HPP_
#define CAFFE_SERIALIZATION_BLOBCODEC_HPP_

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Net;

template <typename Dtype>
class BlobCodec {
 public:
  static shared_ptr<BlobCodec> create_codec(
    const MultinodeParameter& param);

  enum What {
    PARAMS = 0,
    GRADS = 1
  };

  virtual uint32_t encode(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          What what,
                          uint32_t start_element) const = 0;

  virtual bool decode(const BlobUpdate& update,
                      Blob<Dtype>* dest,
                      What what,
                      Dtype alpha,
                      Dtype beta) const = 0;
};

template <typename Dtype>
Dtype check_sum(const Dtype* data, size_t size) {
  Dtype ret = 0.0f;
  for (int i = 0; i < size; ++i) {
    ret += data[i];
  }
  return ret;
}

template <typename Dtype>
Dtype check_sum(Blob<Dtype>* blob, typename BlobCodec<Dtype>::What what) {
  return check_sum(
    ((what == BlobCodec<Dtype>::PARAMS) ?  blob->cpu_data() : blob->cpu_diff()),
    blob->count());
}

}  // namespace caffe

#endif  // CAFFE_SERIALIZATION_BLOBCODEC_HPP_

