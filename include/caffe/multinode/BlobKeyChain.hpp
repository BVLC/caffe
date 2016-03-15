#ifndef CAFFE_BLOBKEYCHAIN_HPP_
#define CAFFE_BLOBKEYCHAIN_HPP_

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
class BlobKeyChain {
 public:
  static shared_ptr<BlobKeyChain> create(size_t layers);

  virtual void lock(int layer_id) = 0;
  virtual bool try_lock(int layer_id) = 0;
  virtual void unlock(int layer_id) = 0;
};

}  // namespace caffe

#endif  // CAFFE_BLOBKEYCHAIN_HPP_

