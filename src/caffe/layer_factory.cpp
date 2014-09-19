#include <string>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {

// GetLayer() defines the overall layer factory. The Get*Layer() functions
// define factories for layers with multiple computational engines.

// Get convolution layer according to engine.
template <typename Dtype>
ConvolutionLayer<Dtype>* GetConvolutionLayer(const string& name,
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
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template ConvolutionLayer<float>* GetConvolutionLayer(const string& name,
    const LayerParameter& param);
template ConvolutionLayer<double>* GetConvolutionLayer(const string& name,
    const LayerParameter& param);

// Get pooling layer according to engine.
template <typename Dtype>
PoolingLayer<Dtype>* GetPoolingLayer(const string& name,
    const LayerParameter& param) {
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
    return new CuDNNPoolingLayer<Dtype>(param);
#endif
  } else {
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template PoolingLayer<float>* GetPoolingLayer(const string& name,
    const LayerParameter& param);
template PoolingLayer<double>* GetPoolingLayer(const string& name,
    const LayerParameter& param);

// Get relu layer according to engine.
template <typename Dtype>
ReLULayer<Dtype>* GetReLULayer(const string& name,
    const LayerParameter& param) {
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
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template ReLULayer<float>* GetReLULayer(const string& name,
    const LayerParameter& param);
template ReLULayer<double>* GetReLULayer(const string& name,
    const LayerParameter& param);

// Get sigmoid layer according to engine.
template <typename Dtype>
SigmoidLayer<Dtype>* GetSigmoidLayer(const string& name,
    const LayerParameter& param) {
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
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template SigmoidLayer<float>* GetSigmoidLayer(const string& name,
    const LayerParameter& param);
template SigmoidLayer<double>* GetSigmoidLayer(const string& name,
    const LayerParameter& param);

// Get tanh layer according to engine.
template <typename Dtype>
TanHLayer<Dtype>* GetTanHLayer(const string& name,
    const LayerParameter& param) {
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
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template TanHLayer<float>* GetTanHLayer(const string& name,
    const LayerParameter& param);
template TanHLayer<double>* GetTanHLayer(const string& name,
    const LayerParameter& param);

// Get softmax layer according to engine.
template <typename Dtype>
SoftmaxLayer<Dtype>* GetSoftmaxLayer(const string& name,
    const LayerParameter& param) {
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
    LOG(FATAL) << "Layer " << name << " has unknown engine.";
  }
}

template SoftmaxLayer<float>* GetSoftmaxLayer(const string& name,
    const LayerParameter& param);
template SoftmaxLayer<double>* GetSoftmaxLayer(const string& name,
    const LayerParameter& param);

// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Layer<Dtype>* GetLayer(const LayerParameter& param) {
  const string& name = param.name();
  const LayerParameter_LayerType& type = param.type();
  switch (type) {
  case LayerParameter_LayerType_ACCURACY:
    return new AccuracyLayer<Dtype>(param);
  case LayerParameter_LayerType_ABSVAL:
    return new AbsValLayer<Dtype>(param);
  case LayerParameter_LayerType_ARGMAX:
    return new ArgMaxLayer<Dtype>(param);
  case LayerParameter_LayerType_BNLL:
    return new BNLLLayer<Dtype>(param);
  case LayerParameter_LayerType_CONCAT:
    return new ConcatLayer<Dtype>(param);
  case LayerParameter_LayerType_CONTRASTIVE_LOSS:
    return new ContrastiveLossLayer<Dtype>(param);
  case LayerParameter_LayerType_CONVOLUTION:
    return GetConvolutionLayer<Dtype>(name, param);
  case LayerParameter_LayerType_DATA:
    return new DataLayer<Dtype>(param);
  case LayerParameter_LayerType_DROPOUT:
    return new DropoutLayer<Dtype>(param);
  case LayerParameter_LayerType_DUMMY_DATA:
    return new DummyDataLayer<Dtype>(param);
  case LayerParameter_LayerType_EUCLIDEAN_LOSS:
    return new EuclideanLossLayer<Dtype>(param);
  case LayerParameter_LayerType_ELTWISE:
    return new EltwiseLayer<Dtype>(param);
  case LayerParameter_LayerType_FLATTEN:
    return new FlattenLayer<Dtype>(param);
  case LayerParameter_LayerType_HDF5_DATA:
    return new HDF5DataLayer<Dtype>(param);
  case LayerParameter_LayerType_HDF5_OUTPUT:
    return new HDF5OutputLayer<Dtype>(param);
  case LayerParameter_LayerType_HINGE_LOSS:
    return new HingeLossLayer<Dtype>(param);
  case LayerParameter_LayerType_IMAGE_DATA:
    return new ImageDataLayer<Dtype>(param);
  case LayerParameter_LayerType_IM2COL:
    return new Im2colLayer<Dtype>(param);
  case LayerParameter_LayerType_INFOGAIN_LOSS:
    return new InfogainLossLayer<Dtype>(param);
  case LayerParameter_LayerType_INNER_PRODUCT:
    return new InnerProductLayer<Dtype>(param);
  case LayerParameter_LayerType_LRN:
    return new LRNLayer<Dtype>(param);
  case LayerParameter_LayerType_MEMORY_DATA:
    return new MemoryDataLayer<Dtype>(param);
  case LayerParameter_LayerType_MVN:
    return new MVNLayer<Dtype>(param);
  case LayerParameter_LayerType_MULTINOMIAL_LOGISTIC_LOSS:
    return new MultinomialLogisticLossLayer<Dtype>(param);
  case LayerParameter_LayerType_POOLING:
    return GetPoolingLayer<Dtype>(name, param);
  case LayerParameter_LayerType_POWER:
    return new PowerLayer<Dtype>(param);
  case LayerParameter_LayerType_RELU:
    return GetReLULayer<Dtype>(name, param);
  case LayerParameter_LayerType_SILENCE:
    return new SilenceLayer<Dtype>(param);
  case LayerParameter_LayerType_SIGMOID:
    return GetSigmoidLayer<Dtype>(name, param);
  case LayerParameter_LayerType_SIGMOID_CROSS_ENTROPY_LOSS:
    return new SigmoidCrossEntropyLossLayer<Dtype>(param);
  case LayerParameter_LayerType_SLICE:
    return new SliceLayer<Dtype>(param);
  case LayerParameter_LayerType_SOFTMAX:
    return GetSoftmaxLayer<Dtype>(name, param);
  case LayerParameter_LayerType_SOFTMAX_LOSS:
    return new SoftmaxWithLossLayer<Dtype>(param);
  case LayerParameter_LayerType_SPLIT:
    return new SplitLayer<Dtype>(param);
  case LayerParameter_LayerType_TANH:
    return GetTanHLayer<Dtype>(name, param);
  case LayerParameter_LayerType_WINDOW_DATA:
    return new WindowDataLayer<Dtype>(param);
  case LayerParameter_LayerType_NONE:
    LOG(FATAL) << "Layer " << name << " has unspecified type.";
  default:
    LOG(FATAL) << "Layer " << name << " has unknown type " << type;
  }
  // just to suppress old compiler warnings.
  return (Layer<Dtype>*)(NULL);
}

template Layer<float>* GetLayer(const LayerParameter& param);
template Layer<double>* GetLayer(const LayerParameter& param);

}  // namespace caffe
