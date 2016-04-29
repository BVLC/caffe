#ifndef CAFFE_SERIALIZATION_BLOBCODEC_HPP_
#define CAFFE_SERIALIZATION_BLOBCODEC_HPP_

#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <algorithm>
#include <numeric>
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
typedef BlobEncoding::What BlobEncodingWhat;

template <typename Dtype>
class BlobCodec {
 public:
  typedef typename BlobEncoding::What What;
  static shared_ptr<BlobCodec> create_codec(
    const MultinodeParameter& param,
    bool ensure_is_single_threaded);

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
double check_sum(const Dtype* data, size_t size) {
  if (size == 0) return 0.0;
  if (size == 1) return *data;
  if (size == 2) return static_cast<double>(data[0]) + data[1];
  return check_sum(data, size / 2)
    + check_sum(data + size / 2, size - size / 2);
}

template <typename Dtype>
double check_sum(Blob<Dtype>* blob, BlobEncodingWhat what) {
  return check_sum(
    ((what == BlobEncoding::PARAMS) ?  blob->cpu_data() : blob->cpu_diff()),
    blob->count());
}

}  // namespace caffe

#endif  // CAFFE_SERIALIZATION_BLOBCODEC_HPP_

