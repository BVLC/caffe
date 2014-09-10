#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

// The layer factory
template <typename Dtype>
class LayerFactory {
 public:
  virtual ~LayerFactory() {};
  virtual Layer<Dtype>* GetLayer(const LayerParameter& param) = 0;
};

template <typename Dtype>
class CaffeLayerFactory : public LayerFactory<Dtype> {
 public:
  virtual ~CaffeLayerFactory() {};
  virtual Layer<Dtype>* GetLayer(const LayerParameter& param);
};

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLayerFactory : public LayerFactory<Dtype> {
 public:
  virtual ~CuDNNLayerFactory() {};
  virtual Layer<Dtype>* GetLayer(const LayerParameter& param);
};
#endif

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
