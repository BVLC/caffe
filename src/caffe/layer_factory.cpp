// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#ifdef MKL2017_SUPPORTED
#include "caffe/layers/mkl_layers.hpp"
#endif
#ifdef MKLDNN_SUPPORTED
#include "caffe/layers/mkldnn_layers.hpp"
#endif
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif

namespace caffe {

// Get convolution layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();
  ConvolutionParameter_Engine engine = conv_param.engine();
#if defined(USE_CUDNN) || defined(USE_MKL2017_AS_DEFAULT_ENGINE) || defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }
#endif
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    if (!use_dilation) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#elif defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    if (!use_dilation) {
      engine = ConvolutionParameter_Engine_MKL2017;
    }
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    if (!use_dilation) {
      engine = ConvolutionParameter_Engine_MKLDNN;
    }
#endif
  }
  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype> >(new CuDNNConvolutionLayer<Dtype>(param));
#endif
#ifdef MKL2017_SUPPORTED
  } else if (engine == ConvolutionParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLConvolutionLayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == ConvolutionParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNConvolutionLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);

// Get inner_product layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetInnerProductLayer(
    const LayerParameter& param) {
  InnerProductParameter ip_param = param.inner_product_param();
  InnerProductParameter_Engine engine = ip_param.engine();
  if (engine == InnerProductParameter_Engine_DEFAULT) {
    engine = InnerProductParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = InnerProductParameter_Engine_CUDNN;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    if (!ip_param.transpose()) {
      engine = InnerProductParameter_Engine_MKLDNN;
    }
#endif
  }
  if (engine == InnerProductParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new InnerProductLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == InnerProductParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNInnerProductLayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == InnerProductParameter_Engine_MKLDNN) {
    if (ip_param.transpose()) {
      LOG(FATAL) << "MKL-DNN doesn't support transposed weights at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype> >(new MKLDNNInnerProductLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(InnerProduct, GetInnerProductLayer);

// Get pooling layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPoolingLayer(const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = PoolingParameter_Engine_CUDNN;
#elif defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    engine = PoolingParameter_Engine_MKL2017;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    PoolingParameter_PoolMethod method = param.pooling_param().pool();
    if (method == PoolingParameter_PoolMethod_MAX)
      engine = PoolingParameter_Engine_MKLDNN;
#endif
  }
  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "cuDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    }
    // CuDNN assumes layers are not being modified in place, thus
    // breaking our index tracking for updates in some cases in Caffe.
    // Until there is a workaround in Caffe (index management) or
    // cuDNN, use Caffe layer to max pooling, or don't use in place
    // layers after max pooling layers
    if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
        return shared_ptr<Layer<Dtype> >(new PoolingLayer<Dtype>(param));
    } else {
        return shared_ptr<Layer<Dtype> >(new CuDNNPoolingLayer<Dtype>(param));
    }
#endif
#ifdef MKL2017_SUPPORTED
  } else if (engine == PoolingParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLPoolingLayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == PoolingParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNPoolingLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);

// Get LRN layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetLRNLayer(const LayerParameter& param) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
    engine = LRNParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = LRNParameter_Engine_CUDNN;
#elif defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    if (param.lrn_param().norm_region()
            == LRNParameter_NormRegion_ACROSS_CHANNELS)
      engine = LRNParameter_Engine_MKL2017;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    if (param.lrn_param().norm_region()
            == LRNParameter_NormRegion_ACROSS_CHANNELS)
      engine = LRNParameter_Engine_MKLDNN;
#endif
  }

  if (engine == LRNParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new LRNLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == LRNParameter_Engine_CUDNN) {
    LRNParameter lrn_param = param.lrn_param();

    if (lrn_param.norm_region() ==LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer<Dtype> >(new CuDNNLCNLayer<Dtype>(param));
    } else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer<Dtype> >(new LRNLayer<Dtype>(param));
      } else {
        return shared_ptr<Layer<Dtype> >(new CuDNNLRNLayer<Dtype>(param));
      }
    }
#endif
#ifdef MKL2017_SUPPORTED
  } else if (engine == LRNParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLLRNLayer<Dtype>(param));
#endif
#if MKLDNN_SUPPORTED
  } else if (engine == LRNParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNLRNLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);

// Get BatchNorm layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetBatchNormLayer(const LayerParameter& param) {
  BatchNormParameter_Engine engine = param.batch_norm_param().engine();

  if (engine == BatchNormParameter_Engine_DEFAULT) {
#if defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    engine = BatchNormParameter_Engine_MKL2017;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    engine = BatchNormParameter_Engine_MKLDNN;
#else
    engine = BatchNormParameter_Engine_CAFFE;
#endif
  }

  if (engine == BatchNormParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new BatchNormLayer<Dtype>(param));
#if defined(MKL2017_SUPPORTED)
  } else if (engine == BatchNormParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLBatchNormLayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == BatchNormParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNBatchNormLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(BatchNorm, GetBatchNormLayer);

// Get Split layer according to engine
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSplitLayer(const LayerParameter& param) {
  SplitParameter_Engine engine = param.split_param().engine();

  if (engine == SplitParameter_Engine_DEFAULT) {
#if defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    engine = SplitParameter_Engine_MKL2017;
#else
    engine = SplitParameter_Engine_CAFFE;
#endif
  }

  if (engine == SplitParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SplitLayer<Dtype>(param));
#if defined(MKL2017_SUPPORTED)
  } else if (engine == SplitParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLSplitLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Split, GetSplitLayer);

// Get ReLU layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetReLULayer(const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#elif defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    engine = ReLUParameter_Engine_MKL2017;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    engine = ReLUParameter_Engine_MKLDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNReLULayer<Dtype>(param));
#endif
#ifdef MKL2017_SUPPORTED
  } else if (engine == ReLUParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLReLULayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == ReLUParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNReLULayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);

// Get concat layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetConcatLayer(const LayerParameter& param) {
  ConcatParameter_Engine engine = param.concat_param().engine();
  if (engine == ConcatParameter_Engine_DEFAULT) {
    engine = ConcatParameter_Engine_CAFFE;
#if defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    if (param.concat_param().axis() == 1)
      engine = ConcatParameter_Engine_MKL2017;
#elif defined(USE_MKLDNN_AS_DEFAULT_ENGINE)
    engine = ConcatParameter_Engine_MKLDNN;
#endif
  }
  if (engine == ConcatParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new ConcatLayer<Dtype>(param));
#if defined(MKL2017_SUPPORTED)
  } else if (engine == ConcatParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLConcatLayer<Dtype>(param));
#endif
#ifdef MKLDNN_SUPPORTED
  } else if (engine == ConcatParameter_Engine_MKLDNN) {
    return shared_ptr<Layer<Dtype> >(new MKLDNNConcatLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknow engine.";
  }
}

REGISTER_LAYER_CREATOR(Concat, GetConcatLayer);

// Get concat layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetEltwiseLayer(const LayerParameter& param) {
  EltwiseParameter_Engine engine = param.eltwise_param().engine();
  if (engine == EltwiseParameter_Engine_DEFAULT) {
    engine = EltwiseParameter_Engine_CAFFE;
#if defined(USE_MKL2017_AS_DEFAULT_ENGINE)
    engine = EltwiseParameter_Engine_MKL2017;
#endif
  }
  if (engine == EltwiseParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new EltwiseLayer<Dtype>(param));
#if defined(MKL2017_SUPPORTED)
  } else if (engine == EltwiseParameter_Engine_MKL2017) {
    return shared_ptr<Layer<Dtype> >(new MKLEltwiseLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknow engine.";
  }
}

REGISTER_LAYER_CREATOR(Eltwise, GetEltwiseLayer);


// Get sigmoid layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSigmoidLayer(const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSigmoidLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);

// Get softmax layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetSoftmaxLayer(const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNSoftmaxLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);

// Get tanh layer according to engine.
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetTanHLayer(const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype> >(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype> >(new CuDNNTanHLayer<Dtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);

#ifdef WITH_PYTHON_LAYER
template <typename Dtype>
shared_ptr<Layer<Dtype> > GetPythonLayer(const LayerParameter& param) {
  Py_Initialize();
  try {
    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<PythonLayer<Dtype> > >(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer);
#endif

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
