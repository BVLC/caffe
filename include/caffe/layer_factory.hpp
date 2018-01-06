/**
 * @brief a layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
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
 * and its type is its c++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
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
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class Layer;

template<typename Dtype, typename MItype, typename MOtype>
class LayerRegistry {
 public:
  typedef shared_ptr<Layer<Dtype, MItype, MOtype> >
            (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype, MItype, MOtype> >
            CreateLayer(const LayerParameter& param);

  static vector<string> LayerTypeList();

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry();

  static string LayerTypeListString();
};

template<typename Dtype, typename MItype, typename MOtype>
class LayerRegisterer {
 public:
  LayerRegisterer(const string& type,
   shared_ptr<Layer<Dtype, MItype, MOtype> > (*creator)(const LayerParameter&));
};


}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
