#include <vector>
#include "boost/make_shared.hpp"
#include "boost/thread.hpp"
#include "caffe/multinode/BlobKeyChain.hpp"

namespace caffe {


namespace {

template <typename Dtype, bool IsStub>
struct BlobKeyChainImpl : public BlobKeyChain<Dtype> {
  std::vector<shared_ptr<boost::recursive_mutex> > mtxs;

  explicit BlobKeyChainImpl(size_t layers)
    : mtxs(layers) {
    for (int i = 0; i < mtxs.size(); ++i) {
      mtxs[i].reset(new boost::recursive_mutex());
    }
  }

  virtual void lock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    if (!IsStub) mtxs[layer_id]->lock();
  }
  virtual void unlock(int layer_id) {
    CHECK_GE(layer_id, 0);
    CHECK(layer_id < mtxs.size());
    if (!IsStub) mtxs[layer_id]->unlock();
  }

  virtual void lock(int layer_id, int blob_id, int part) {
    if (!IsStub) lock(layer_id);
  }
  virtual void unlock(int layer_id, int blob_id, int part) {
    if (!IsStub) unlock(layer_id);
  }
};
}  // namespace

template <typename Dtype>
shared_ptr<BlobKeyChain<Dtype> >
  BlobKeyChain<Dtype>::create(size_t layers) {
  return boost::make_shared<BlobKeyChainImpl<Dtype, false> >(layers);
}

template <typename Dtype>
shared_ptr<BlobKeyChain<Dtype> >
  BlobKeyChain<Dtype>::create_empty(size_t layers) {
  return boost::make_shared<BlobKeyChainImpl<Dtype, true> >(layers);
}

INSTANTIATE_CLASS(BlobKeyChain);
}  // namespace caffe

