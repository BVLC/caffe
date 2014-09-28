// This file defines a layer factory that allows one to register layers. During
// runtime, registered layers could be called by passing a LayerParameter:
//     LayerRegistry<Dtype>::CreateLayer(param);
//
// There are two ways to register a layer. Assuming that we have a layer like:
//
//   template <typename Dtype>
//   class MyAwesomeLayer : public Layer<Dtype> {
//     // your implementations
//   };
//
// and its type is defined in the protobuffer as
//
//   enum LayerType {
//     // other definitions
//     AWESOME = 46,
//   }
//
// If the layer is going to be created simply by its constructor, in your c++
// file, add the following line:
//
//    REGISTER_LAYER_CLASS(AWESOME, MyAwesomeLayer);
//
// Or, if the layer is going to be created by another creator function, in the
// format of:
//
//    template <typename Dtype>
//    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
//      // your implementation
//    }
//
// (for example, when your layer has multiple backends, see GetConvolutionLayer
// for a use case), then you can register the creator function instead, like
//
// REGISTER_LAYER_CREATOR(AWESOME, GetMyAwesomeLayer)
//
// Note that each layer type should only be registered once.

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  typedef Layer<Dtype>* (*Creator)(const LayerParameter&);
  typedef std::map<LayerParameter_LayerType, Creator> CreatorRegistry;

  // Adds a creator.
  static void AddCreator(const LayerParameter_LayerType& type,
                         Creator creator) {
    CHECK_EQ(registry_.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry_[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static Layer<Dtype>* CreateLayer(const LayerParameter& param) {
    LOG(INFO) << "Creating layer " << param.name();
    const LayerParameter_LayerType& type = param.type();
    CHECK_EQ(registry_.count(type), 1);
    return registry_[type](param);
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}
  static CreatorRegistry registry_;
};

// Static variables for the templated layer factory registry.
template <typename Dtype>
typename LayerRegistry<Dtype>::CreatorRegistry LayerRegistry<Dtype>::registry_;

template <typename Dtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const LayerParameter_LayerType& type,
                  Layer<Dtype>* (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};


#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(                            \
      LayerParameter_LayerType_##type, creator<float>);                        \
  static LayerRegisterer<double> g_creator_d_##type(                           \
      LayerParameter_LayerType_##type, creator<double>)

#define REGISTER_LAYER_CLASS(type, clsname)                                    \
  template <typename Dtype>                                                    \
  Layer<Dtype>* Creator_##clsname(const LayerParameter& param) {               \
    return new clsname<Dtype>(param);                                          \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##clsname)

// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
// Yangqing's note: With LayerRegistry, we no longer need this thin wrapper any
// more. It is provided here for backward compatibility and should be removed in
// the future.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  return LayerRegistry<Dtype>::CreateLayer(param);
}

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
