#ifdef USE_LIBDNN
#ifndef CAFFE_LIBDNN_POOL_LAYER_HPP_
#define CAFFE_LIBDNN_POOL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/pooling_layer.hpp"

#include "caffe/greentea/libdnn.hpp"

namespace caffe {

template <typename Dtype>
class LibDNNPoolingLayer : public PoolingLayer<Dtype> {
 public:
  explicit LibDNNPoolingLayer(const LayerParameter& param)
      : PoolingLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual ~LibDNNPoolingLayer();


 protected:
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


 private:
  shared_ptr<LibDNNPool<Dtype> > libdnn_;
};

}  // namespace caffe

#endif  // CAFFE_LIBDNN_POOL_LAYER_HPP_
#endif  // USE_LIBDNN
