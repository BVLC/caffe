#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {

#define GET_LAYER(Type, Class) \
  virtual inline Layer<Dtype>* Get##Type##Layer(const LayerParameter& \
                                                param) { \
    return new Class##Layer<Dtype>(param); \
  }

template <typename Dtype>
class CaffeLayerFactory {
 public:
  virtual ~CaffeLayerFactory() {}
  virtual Layer<Dtype>* GetLayer(const LayerParameter& param);
  GET_LAYER(Accuracy, Accuracy)
  GET_LAYER(AbsVal, AbsVal)
  GET_LAYER(ArgMax, ArgMax)
  GET_LAYER(BNLL, BNLL)
  GET_LAYER(Concat, Concat)
  GET_LAYER(Convolution, Convolution)
  GET_LAYER(Data, Data)
  GET_LAYER(Dropout, Dropout)
  GET_LAYER(DummyData, DummyData)
  GET_LAYER(EuclideanLoss, EuclideanLoss)
  GET_LAYER(Eltwise, Eltwise)
  GET_LAYER(Flatten, Flatten)
  GET_LAYER(HDF5Data, HDF5Data)
  GET_LAYER(HDF5Output, HDF5Output)
  GET_LAYER(HingeLoss, HingeLoss)
  GET_LAYER(ImageData, ImageData)
  GET_LAYER(Im2col, Im2col)
  GET_LAYER(InfogainLoss, InfogainLoss)
  GET_LAYER(InnerProduct, InnerProduct)
  GET_LAYER(LRN, LRN)
  GET_LAYER(MemoryData, MemoryData)
  GET_LAYER(MVN, MVN)
  GET_LAYER(MultinomialLogisticLoss, MultinomialLogisticLoss)
  GET_LAYER(Pooling, Pooling)
  GET_LAYER(Power, Power)
  GET_LAYER(ReLU, ReLU)
  GET_LAYER(Silence, Silence)
  GET_LAYER(Sigmoid, Sigmoid)
  GET_LAYER(SigmoidCrossEntropyLoss, SigmoidCrossEntropyLoss)
  GET_LAYER(Slice, Slice)
  GET_LAYER(Softmax, Softmax)
  GET_LAYER(SoftmaxWithLoss, SoftmaxWithLoss)
  GET_LAYER(Split, Split)
  GET_LAYER(TanH, TanH)
  GET_LAYER(WindowData, WindowData)
};

#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNLayerFactory : public CaffeLayerFactory<Dtype> {
 public:
  virtual ~CuDNNLayerFactory() {}
  virtual Layer<Dtype>* GetLayer(const LayerParameter& param);
  GET_LAYER(Convolution, CuDNNConvolution)
  GET_LAYER(Pooling, CuDNNPooling)
  GET_LAYER(ReLU, CuDNNReLU)
  GET_LAYER(Sigmoid, CuDNNSigmoid)
  GET_LAYER(Softmax, CuDNNSoftmax)
  GET_LAYER(TanH, CuDNNTanH)
};
#endif

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
