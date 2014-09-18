#include <string>

#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {

// GetLayer() defines the overall layer factory. The *LayerFactory classes
// define factories for layers of different computational engines.

#define LAYER_CASE(TYPE, Type) \
  case LayerParameter_LayerType_##TYPE: \
    return Get##Type##Layer(param)

template <typename Dtype>
Layer<Dtype>* CaffeLayerFactory<Dtype>::GetLayer(const LayerParameter&
                                                 param) {
  const string& name = param.name();
  const LayerParameter_LayerType& type = param.type();
  switch (type) {
  LAYER_CASE(ACCURACY, Accuracy);
  LAYER_CASE(ABSVAL, AbsVal);
  LAYER_CASE(ARGMAX, ArgMax);
  LAYER_CASE(BNLL, BNLL);
  LAYER_CASE(CONCAT, Concat);
  LAYER_CASE(CONVOLUTION, Convolution);
  LAYER_CASE(DATA, Data);
  LAYER_CASE(DROPOUT, Dropout);
  LAYER_CASE(DUMMY_DATA, DummyData);
  LAYER_CASE(EUCLIDEAN_LOSS, EuclideanLoss);
  LAYER_CASE(ELTWISE, Eltwise);
  LAYER_CASE(FLATTEN, Flatten);
  LAYER_CASE(HDF5_DATA, HDF5Data);
  LAYER_CASE(HDF5_OUTPUT, HDF5Output);
  LAYER_CASE(HINGE_LOSS, HingeLoss);
  LAYER_CASE(IMAGE_DATA, ImageData);
  LAYER_CASE(IM2COL, Im2col);
  LAYER_CASE(INFOGAIN_LOSS, InfogainLoss);
  LAYER_CASE(INNER_PRODUCT, InnerProduct);
  LAYER_CASE(LRN, LRN);
  LAYER_CASE(MEMORY_DATA, MemoryData);
  LAYER_CASE(MVN, MVN);
  LAYER_CASE(MULTINOMIAL_LOGISTIC_LOSS, MultinomialLogisticLoss);
  LAYER_CASE(POOLING, Pooling);
  LAYER_CASE(POWER, Power);
  LAYER_CASE(RELU, ReLU);
  LAYER_CASE(SILENCE, Silence);
  LAYER_CASE(SIGMOID, Sigmoid);
  LAYER_CASE(SIGMOID_CROSS_ENTROPY_LOSS, SigmoidCrossEntropyLoss);
  LAYER_CASE(SLICE, Slice);
  LAYER_CASE(SOFTMAX, Softmax);
  LAYER_CASE(SOFTMAX_LOSS, SoftmaxWithLoss);
  LAYER_CASE(SPLIT, Split);
  LAYER_CASE(TANH, TanH);
  LAYER_CASE(WINDOW_DATA, WindowData);
  case LayerParameter_LayerType_NONE:
    LOG(FATAL) << "Layer " << name << " has unspecified type.";
  default:
    LOG(FATAL) << "Layer " << name << " has unknown type " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

INSTANTIATE_CLASS(CaffeLayerFactory);

#ifdef USE_CUDNN
INSTANTIATE_CLASS(CuDNNLayerFactory);
#endif

}  // namespace caffe
