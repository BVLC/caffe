#ifdef USE_LIBDNN
#ifndef CAFFE_LIBDNN_POOL_LAYER_HPP_
#define CAFFE_LIBDNN_POOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/libdnn/libdnn_pool.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class LibDNNPoolingLayer : public PoolingLayer<Dtype, MItype, MOtype> {
 public:
  explicit LibDNNPoolingLayer(const LayerParameter& param)
      : PoolingLayer<Dtype, MItype, MOtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual ~LibDNNPoolingLayer();

 protected:
  virtual void Forward_gpu(
      const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top);
  virtual void Backward_gpu(
      const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom);

 private:
  shared_ptr<LibDNNPool<MItype, MOtype> > libdnn_;
};

}  // namespace caffe

#endif  // CAFFE_LIBDNN_POOL_LAYER_HPP_
#endif  // USE_LIBDNN
