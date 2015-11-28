/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
<<<<<<< HEAD
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
=======
<<<<<<< HEAD
<<<<<<< HEAD
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
=======
 * and its type is defined in the protobuffer as
 *
 *   enum LayerType {
 *     // other definitions
 *     AWESOME = 46,
 *   }
>>>>>>> origin/BVLC/parallel
=======
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
<<<<<<< HEAD
 *    REGISTER_LAYER_CLASS(MyAwesome);
=======
<<<<<<< HEAD
<<<<<<< HEAD
 *    REGISTER_LAYER_CLASS(MyAwesome);
=======
 *    REGISTER_LAYER_CLASS(AWESOME, MyAwesomeLayer);
>>>>>>> origin/BVLC/parallel
=======
 *    REGISTER_LAYER_CLASS(MyAwesome);
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
<<<<<<< HEAD
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
=======
<<<<<<< HEAD
<<<<<<< HEAD
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
=======
 * REGISTER_LAYER_CREATOR(AWESOME, GetMyAwesomeLayer)
>>>>>>> origin/BVLC/parallel
=======
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
<<<<<<< HEAD
#include <string>
#include <vector>
=======
<<<<<<< HEAD
<<<<<<< HEAD
#include <string>
#include <vector>
=======
>>>>>>> origin/BVLC/parallel
=======
#include <string>
#include <vector>
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
<<<<<<< HEAD
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;
=======
  typedef Layer<Dtype>* (*Creator)(const LayerParameter&);
  typedef std::map<LayerParameter_LayerType, Creator> CreatorRegistry;
>>>>>>> origin/BVLC/parallel
=======
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
<<<<<<< HEAD
  static void AddCreator(const string& type, Creator creator) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
  static void AddCreator(const string& type, Creator creator) {
=======
  static void AddCreator(const LayerParameter_LayerType& type,
                         Creator creator) {
>>>>>>> origin/BVLC/parallel
=======
  static void AddCreator(const string& type, Creator creator) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
  static Layer<Dtype>* CreateLayer(const LayerParameter& param) {
    LOG(INFO) << "Creating layer " << param.name();
    const LayerParameter_LayerType& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1);
    return registry[type](param);
  }

>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() {}
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
};


template <typename Dtype>
class LayerRegisterer {
 public:
<<<<<<< HEAD
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
=======
  LayerRegisterer(const LayerParameter_LayerType& type,
                  Layer<Dtype>* (*creator)(const LayerParameter&)) {
>>>>>>> origin/BVLC/parallel
=======
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};


#define REGISTER_LAYER_CREATOR(type, creator)                                  \
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
<<<<<<< HEAD
=======
<<<<<<< HEAD
=======
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
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
