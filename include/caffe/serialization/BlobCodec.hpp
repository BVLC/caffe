#ifndef CAFFE_SERIALIZATION_BLOBCODEC_HPP_
#define CAFFE_SERIALIZATION_BLOBCODEC_HPP_

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Net;

struct BlobEncoding {
  enum What {
    PARAMS = 0,
    GRADS = 1
  };
};
typedef typename BlobEncoding::What BlobEncodingWhat;

template <typename Dtype>
class BlobCodec {
 public:
  typedef typename BlobEncoding::What What;
  static shared_ptr<BlobCodec> create_codec(
    const MultinodeParameter& param);

  virtual uint32_t encode(BlobUpdate* msg,
                          const Blob<Dtype>* src,
                          What what,
                          uint32_t part) const = 0;

  virtual bool decode(const BlobUpdate& update,
                      Blob<Dtype>* dest,
                      What what,
                      Dtype alpha,
                      Dtype beta) const = 0;

  virtual size_t max_elements_per_part() const = 0;
  virtual size_t packet_size() const = 0;
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
Dtype check_sum(Blob<Dtype>* blob, BlobEncodingWhat what) {
  return check_sum(
    ((what == BlobEncoding::PARAMS) ?  blob->cpu_data() : blob->cpu_diff()),
    blob->count());
}

}  // namespace caffe

#endif  // CAFFE_SERIALIZATION_BLOBCODEC_HPP_

