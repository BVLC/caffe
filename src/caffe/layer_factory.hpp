// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_FACTORY_HPP_
#define CAFFE_LAYER_FACTORY_HPP_

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/layer_param.pb.h"


namespace caffe {


// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const std::string& type = param.type();
  if (type == "conv") {
    return new ConvolutionLayer<Dtype>(param);
  } else if (type == "dropout") {
    return new DropoutLayer<Dtype>(param);
  } else if (type == "im2col") {
    return new Im2colLayer<Dtype>(param);
  } else if (type == "innerproduct") {
    return new InnerProductLayer<Dtype>(param);
  } else if (type == "lrn") {
    return new LRNLayer<Dtype>(param);
  } else if (type == "padding") {
    return new PaddingLayer<Dtype>(param);
  } else if (type == "pool") {
    return new PoolingLayer<Dtype>(param);
  } else if (type == "relu") {
    return new ReLULayer<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown filler name: " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}


}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_HPP_
