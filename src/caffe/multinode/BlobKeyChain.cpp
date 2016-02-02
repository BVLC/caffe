#include <vector>
#include "boost/make_shared.hpp"
#include "boost/thread.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"

namespace caffe {


namespace {

template <typename Dtype>
struct BlobKeyChainImpl : public BlobKeyChain<Dtype> {
  std::vector<shared_ptr<boost::mutex> > mtxs;

  explicit BlobKeyChainImpl(size_t layers)
    : mtxs(layers) {
    for (int i = 0; i < mtxs.size(); ++i) {
      mtxs[i].reset(new boost::mutex());
    }
  }

  virtual void lock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    mtxs[layer_id]->lock();
  }
  virtual bool try_lock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    return mtxs[layer_id]->try_lock();
  }
  virtual void unlock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    mtxs[layer_id]->unlock();
  }
};
}  // namespace

template <typename Dtype>
shared_ptr<BlobKeyChain<Dtype> > BlobKeyChain<Dtype>::create(size_t layers) {
  return boost::make_shared<BlobKeyChainImpl<Dtype> >(layers);
}

INSTANTIATE_CLASS(BlobKeyChain);
}  // namespace caffe

