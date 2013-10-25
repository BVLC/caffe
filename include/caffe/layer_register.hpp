// This function implements a registry keeping a record of all layer names.
// Copyright Yangqing Jia 2013

#ifndef CAFFE_LAYER_REGISTER_HPP_
#define CAFFE_LAYER_REGISTER_HPP_

#include <string>
#include <map>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

// Internal: the layer registry
template <typename Dtype>
class LayerRegistry {
 public:
  typedef Layer<Dtype>* (*Creator)(const LayerParameter&);

  LayerRegistry() : layer_map_() {}
  ~LayerRegistry() {}

  void AddCreator(string name, Creator creator) {
    layer_map_[name] = creator;
  }

  inline Layer<Dtype>* CreateLayer(const string& name, const LayerParameter& param) {
    typename LayerMap::const_iterator it = layer_map_.find(name);
    if (it == layer_map_.end()) {
      LOG(FATAL) << "Unknown layer: " << name;
    }
    return *(it->second)(param);
  }

 private:
  typedef typename std::map<string, Creator> LayerMap;
  LayerMap layer_map_;
};


// Internal: the function to get the layer registry.
template <typename Dtype>
inline LayerRegistry<Dtype>& GetLayerRegistry() {
  static LayerRegistry<Dtype> registry;
  return registry;
};


// Internal: The registerer class to register a class.
template <typename Dtype>
class LayerCreatorRegisterer {
 public:
  explicit LayerCreatorRegisterer(const string& name,
      typename LayerRegistry<Dtype>::Creator creator) {
    GetLayerRegistry<Dtype>().AddCreator(name, creator);
  }
  ~LayerCreatorRegisterer() {}
};


// The macro to use for register a layer. For example, if you have a
// ConvolutionLayer and want to register it with name "conv", do
//    REGISTER_LAYER("conv", ConvolutionLayer)
#define REGISTER_LAYER(name, DerivedLayer) \
  template <typename Dtype> \
  Layer<Dtype>* Create##DerivedLayer(const LayerParameter& param) { \
    return new DerivedLayer<Dtype>(param); \
  } \
  LayerCreatorRegisterer<float> g_creator_float_##DerivedLayer( \
      name, &Create##DerivedLayer<float>); \
  LayerCreatorRegisterer<double> g_creator_double_##DerivedLayer( \
      name, &Create##DerivedLayer<double>)


// The function to call to get a layer.
template <typename Dtype>
Layer<Dtype>* CreateLayer(const LayerParameter& param) {
  return GetLayerRegistry<Dtype>().CreateLayer(param.type(), param);
}

}  // namespace caffe

# endif  // CAFFE_LAYER_REGISTER_HPP_
