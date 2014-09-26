#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {

// GetLayer() defines the overall layer factory. The Get*Layer() functions
// define factories for layers with multiple computational engines.

// Get convolution layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ConvolutionParameter_Engine_CUDNN;
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return new ConvolutionLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    return new CuDNNConvolutionLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Get pooling layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return new PoolingLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    if (p_param.pad_h() || p_param.pad_w() || param.top_size() > 1) {
      LOG(INFO) << "CUDNN does not support padding or multiple tops. "
                << "Using Caffe's own pooling layer.";
      return new PoolingLayer<DType>(param);
    }
    return new CuDNNPoolingLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Get relu layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return new ReLULayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return new CuDNNReLULayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Get sigmoid layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return new SigmoidLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return new CuDNNSigmoidLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Get tanh layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return new TanHLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return new CuDNNTanHLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Get softmax layer according to engine.
template <typename Dtype>
Layer<Dtype>* GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return new SoftmaxLayer<Dtype>(param);
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return new CuDNNSoftmaxLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

// Layers that have a specific creator function.
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_CONVOLUTION,
                       GetConvolutionLayer, ConvolutionLayer);
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_POOLING,
                       GetPoolingLayer, PoolingLayer);
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_RELU,
                       GetReLULayer, ReLULayer);
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_SIGMOID,
                       GetSigmoidLayer, SigmoidLayer);
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_SOFTMAX,
                       GetSoftmaxLayer, SoftmaxLayer);
REGISTER_LAYER_CREATOR(LayerParameter_LayerType_TANH,
                       GetTanHLayer, TanHLayer);
// Layers that use their constructor as their default creator.
REGISTER_LAYER_CLASS(LayerParameter_LayerType_ACCURACY, AccuracyLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_ABSVAL, AbsValLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_ARGMAX, ArgMaxLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_BNLL, BNLLLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_CONCAT, ConcatLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_CONTRASTIVE_LOSS,
                     ContrastiveLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_DATA, DataLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_DROPOUT, DropoutLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_DUMMY_DATA, DummyDataLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_EUCLIDEAN_LOSS,
                     EuclideanLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_ELTWISE, EltwiseLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_FLATTEN, FlattenLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_HDF5_DATA, HDF5DataLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_HDF5_OUTPUT, HDF5OutputLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_HINGE_LOSS, HingeLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_IMAGE_DATA, ImageDataLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_IM2COL, Im2colLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_INFOGAIN_LOSS, InfogainLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_INNER_PRODUCT, InnerProductLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_LRN, LRNLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_MEMORY_DATA, MemoryDataLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_MVN, MVNLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS,
                     MultinomialLogisticLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_POWER, PowerLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_SILENCE, SilenceLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS,
                     SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_SLICE, SliceLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_SOFTMAX_LOSS,
                     SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_SPLIT, SplitLayer);
REGISTER_LAYER_CLASS(LayerParameter_LayerType_WINDOW_DATA, WindowDataLayer);

}  // namespace caffe
