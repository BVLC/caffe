#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {

// GetLayer() defines the overall layer factory. The *LayerFactory classes
// define factories for layers of different computational engines.

// A function to get a specific layer from the specification given in
// LayerParameter. This implementation uses the factory pattern.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const string& name = param.name();
  const LayerParameter_Engine& engine = param.engine();
  CaffeLayerFactory<Dtype>* caffe_layer_factory =
      new CaffeLayerFactory<Dtype>();
#ifdef USE_CUDNN
  CaffeLayerFactory<Dtype>* cudnn_layer_factory =
      new CuDNNLayerFactory<Dtype>();
#endif
  switch (engine) {
  case LayerParameter_Engine_CAFFE:
    return caffe_layer_factory->GetLayer(param);
#ifdef USE_CUDNN
  case LayerParameter_Engine_CUDNN:
    return cudnn_layer_factory->GetLayer(param);
#endif
  case LayerParameter_Engine_DEFAULT:
#ifdef USE_CUDNN
    return cudnn_layer_factory->GetLayer(param);
#else
    return caffe_layer_factory->GetLayer(param);
#endif
  default:
    LOG(FATAL) << "Layer " << name << " has unknown engine " << engine;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

template Layer<float>* GetLayer(const LayerParameter& param);
template Layer<double>* GetLayer(const LayerParameter& param);

}  // namespace caffe
