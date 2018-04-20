// Make sure we include Python.h before any system header
// to avoid _POSIX_C_SOURCE redefinition
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/conv_fft_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/conv_spatial_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/util/type_utils.hpp"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_deconv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#ifdef USE_LIBDNN
#include "caffe/layers/libdnn_conv_layer.hpp"
#include "caffe/layers/libdnn_deconv_layer.hpp"
#include "caffe/layers/libdnn_pool_layer.hpp"
#endif  // USE_LIBDNN

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif  // WITH_PYTHON_LAYER

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
typename LayerRegistry<Dtype, MItype, MOtype>::CreatorRegistry&
LayerRegistry<Dtype, MItype, MOtype>::Registry() {
  static CreatorRegistry* g_registry_ = new CreatorRegistry();
  return *g_registry_;
}

// Adds a creator.
template <typename Dtype, typename MItype, typename MOtype>
void LayerRegistry<Dtype, MItype, MOtype>::AddCreator(const string& type,
                                                      Creator creator) {
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 0) << "Layer type " << type
                                    << " already registered"
                                    << " for data types ("
                                    << safe_type_name<Dtype>() << ","
                                    << safe_type_name<MItype>() << ","
                                    << safe_type_name<MOtype>() << ").";
  registry[type] = creator;
}

// Get a layer using a LayerParameter.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> >
  LayerRegistry<Dtype, MItype, MOtype>::CreateLayer(
    const LayerParameter& param) {
  if (Caffe::root_solver()) {
    LOG(INFO) << "Creating layer " << param.name();
  }
  const string& type = param.type();
  CreatorRegistry& registry = Registry();
  CHECK_EQ(registry.count(type), 1)
      << "Unknown layer type: " << type
      << " for data types ("
      << safe_type_name<Dtype>() << ","
      << safe_type_name<MItype>() << ","
      << safe_type_name<MOtype>() << ")"
      << " (known types: " << LayerTypeListString() << ")";
  return registry[type](param);
}

template <typename Dtype, typename MItype, typename MOtype>
vector<string> LayerRegistry<Dtype, MItype, MOtype>::LayerTypeList() {
  CreatorRegistry& registry = Registry();
  vector<string> layer_types;
  for (typename CreatorRegistry::iterator iter = registry.begin();
       iter != registry.end(); ++iter) {
    layer_types.push_back(iter->first);
  }
  return layer_types;
}

// Layer registry should never be instantiated - everything is done with its
// static variables.
template <typename Dtype, typename MItype, typename MOtype>
LayerRegistry<Dtype, MItype, MOtype>::LayerRegistry() {}

template <typename Dtype, typename MItype, typename MOtype>
string LayerRegistry<Dtype, MItype, MOtype>::LayerTypeListString() {
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

template <typename Dtype, typename MItype, typename MOtype>
LayerRegisterer<Dtype, MItype, MOtype>::LayerRegisterer(const string& type,
  shared_ptr<Layer<Dtype, MItype, MOtype> > (*creator)(const LayerParameter&)) {
  // LOG(INFO) << "Registering layer type: " << type;
  LayerRegistry<Dtype, MItype, MOtype>::AddCreator(type, creator);
}

INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (half_fp), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (float), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (double), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (uint8_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (uint16_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (uint32_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegistry,
                             (uint64_t), PROTO_TYPES, PROTO_TYPES);

INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (half_fp), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (float), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (double), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (uint8_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (uint16_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (uint32_t), PROTO_TYPES, PROTO_TYPES);
INSTANTIATE_CLASS_3T_GUARDED(LayerRegisterer,
                             (uint64_t), PROTO_TYPES, PROTO_TYPES);

bool checkConvolutionDilated(ConvolutionParameter param) {
  for (int i = 0; i < param.dilation_size(); ++i) {
    if (param.dilation(i) > 1) {
      return true;
    }
  }
  return false;
}

bool checkPoolingDilated(PoolingParameter param) {
  for (int i = 0; i < param.dilation_size(); ++i) {
    if (param.dilation(i) > 1) {
      return true;
    }
  }
  return false;
}

// Get convolution layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetConvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;

#ifdef USE_LIBDNN
    if (!(Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CPU)) {
      engine = ConvolutionParameter_Engine_LIBDNN;
    }
#endif

#ifdef USE_CUDNN
    if (Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CUDA) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#endif

#ifdef USE_INTEL_SPATIAL
    if (Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
      if (Caffe::GetDevice(param.device(), true)->CheckVendor("Intel")
          && Caffe::GetDevice(param.device(), true)->CheckType("GPU")) {
        engine = ConvolutionParameter_Engine_INTEL_SPATIAL;
      }
    }
#endif  // USE_INTEL_SPATIAL
  }

#ifdef USE_INTEL_SPATIAL
  if (engine == ConvolutionParameter_Engine_INTEL_SPATIAL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >
             (new ConvolutionLayerSpatial<Dtype>(param));
  }
#endif  // USE_INTEL_SPATIAL
#ifdef USE_FFT
  if (engine == ConvolutionParameter_Engine_FFT) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >
             (new ConvolutionLayerFFT<Dtype>(param));
  }
#endif  // USE_FFT

  if (engine == ConvolutionParameter_Engine_CUDNN
      && (Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL
          || checkConvolutionDilated(param.convolution_param()))) {
    engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_LIBDNN
    engine = ConvolutionParameter_Engine_LIBDNN;
#endif
  }

  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new ConvolutionLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN) {
    if (checkConvolutionDilated(param.convolution_param())) {
      LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new CuDNNConvolutionLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_CUDNN
#ifdef USE_LIBDNN
  } else if (engine == ConvolutionParameter_Engine_LIBDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new LibDNNConvolutionLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_LIBDNN
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer,
                       (double), (double), (double));

// Get lower precision convolution layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetConvolutionLayerLowerPrecision(
    const LayerParameter& param) {
  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;

#ifdef USE_LIBDNN
    if (!(Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CPU)) {
      engine = ConvolutionParameter_Engine_LIBDNN;
    }
#endif
  }

  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new ConvolutionLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_LIBDNN
  } else if (engine == ConvolutionParameter_Engine_LIBDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new LibDNNConvolutionLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_LIBDNN
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayerLowerPrecision,
                       (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayerLowerPrecision,
                       (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayerLowerPrecision,
                       (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayerLowerPrecision,
                       (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayerLowerPrecision,
                       (uint64_t), (uint64_t), (uint64_t));


// Get deconvolution layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetDeconvolutionLayer(
    const LayerParameter& param) {
  ConvolutionParameter conv_param = param.convolution_param();

  bool use_dilation = false;
  for (int i = 0; i < conv_param.dilation_size(); ++i) {
    if (conv_param.dilation(i) > 1) {
      use_dilation = true;
    }
  }

  ConvolutionParameter_Engine engine = param.convolution_param().engine();
  if (engine == ConvolutionParameter_Engine_DEFAULT) {
    engine = ConvolutionParameter_Engine_CAFFE;

#ifdef USE_LIBDNN
    engine = ConvolutionParameter_Engine_LIBDNN;
#endif

#ifdef USE_CUDNN
    if (Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CUDA
        && !use_dilation) {
      engine = ConvolutionParameter_Engine_CUDNN;
    }
#endif
  }

  if (engine == ConvolutionParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(new
                              DeconvolutionLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ConvolutionParameter_Engine_CUDNN){
    if (use_dilation) {
      LOG(FATAL) << "CuDNN doesn't support the dilated deconvolution at Layer "
                 << param.name();
    }
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(new
                         CuDNNDeconvolutionLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_CUDNN
#ifdef USE_LIBDNN
  } else if (engine == ConvolutionParameter_Engine_LIBDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(new
                        LibDNNDeconvolutionLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_LIBDNN
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Deconvolution, GetDeconvolutionLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(Deconvolution, GetDeconvolutionLayer,
                       (double), (double), (double));


// Get pooling layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetPoolingLayer(
    const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_LIBDNN
    if (!(Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CPU)) {
      engine = PoolingParameter_Engine_LIBDNN;
    }
#endif
  }
  if (engine == PoolingParameter_Engine_LIBDNN) {
#ifdef USE_LIBDNN
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new LibDNNPoolingLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_LIBDNN
  } else if (engine == PoolingParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL
      || checkPoolingDilated(param.pooling_param())) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new PoolingLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == PoolingParameter_Engine_CUDNN) {
    if (param.top_size() > 1) {
      LOG(INFO) << "cuDNN does not support multiple tops. "
                << "Using Caffe's own pooling layer.";
      return shared_ptr<Layer<Dtype, MItype, MOtype> >(
          new PoolingLayer<Dtype, MItype, MOtype>(param));
    }
    if (checkPoolingDilated(param.pooling_param())) {
      LOG(FATAL) << "CuDNN doesn't support the dilated pooling at Layer "
                 << param.name();
      return shared_ptr<Layer<Dtype, MItype, MOtype> >(
          new PoolingLayer<Dtype, MItype, MOtype>(param));
    }
    // CuDNN assumes layers are not being modified in place, thus
    // breaking our index tracking for updates in some cases in Caffe.
    // Until there is a workaround in Caffe (index management) or
    // cuDNN, use Caffe layer to max pooling, or don't use in place
    // layers after max pooling layers
    if (param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
        return shared_ptr<Layer<Dtype, MItype, MOtype> >(
            new PoolingLayer<Dtype, MItype, MOtype>(param));
    } else {
        return shared_ptr<Layer<Dtype, MItype, MOtype> >(
            new CuDNNPoolingLayer<Dtype, MItype, MOtype>(param));
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer,
                       (double), (double), (double));


// Get lower precision pooling layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetPoolingLayerLowerPrecision(
    const LayerParameter& param) {
  PoolingParameter_Engine engine = param.pooling_param().engine();
  if (engine == PoolingParameter_Engine_DEFAULT) {
    engine = PoolingParameter_Engine_CAFFE;

#ifdef USE_LIBDNN
    if (!(Caffe::GetDevice(param.device(), true)->backend() == BACKEND_CPU)) {
      engine = PoolingParameter_Engine_LIBDNN;
    }
#endif
  }

  if (engine == PoolingParameter_Engine_CAFFE) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new PoolingLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_LIBDNN
  } else if (engine == PoolingParameter_Engine_LIBDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new LibDNNPoolingLayer<Dtype, MItype, MOtype>(param));
#endif  // USE_LIBDNN
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayerLowerPrecision,
                       (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayerLowerPrecision,
                       (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayerLowerPrecision,
                       (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayerLowerPrecision,
                       (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayerLowerPrecision,
                       (uint64_t), (uint64_t), (uint64_t));




// Get LRN layer according to engine
template <typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetLRNLayer(
    const LayerParameter& param) {
  LRNParameter_Engine engine = param.lrn_param().engine();

  if (engine == LRNParameter_Engine_DEFAULT) {
    engine = LRNParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = LRNParameter_Engine_CUDNN;
#endif
  }

  if (engine == LRNParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new LRNLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == LRNParameter_Engine_CUDNN) {
    LRNParameter lrn_param = param.lrn_param();

    if (lrn_param.norm_region() ==LRNParameter_NormRegion_WITHIN_CHANNEL) {
      return shared_ptr<Layer<Dtype, MItype, MOtype> >(
          new CuDNNLCNLayer<Dtype, MItype, MOtype>(param));
    } else {
      // local size is too big to be handled through cuDNN
      if (param.lrn_param().local_size() > CUDNN_LRN_MAX_N) {
        return shared_ptr<Layer<Dtype, MItype, MOtype> >(
            new LRNLayer<Dtype, MItype, MOtype>(param));
      } else {
        return shared_ptr<Layer<Dtype, MItype, MOtype> >(
            new CuDNNLRNLayer<Dtype, MItype, MOtype>(param));
      }
    }
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(LRN, GetLRNLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(LRN, GetLRNLayer,
                       (double), (double), (double));

// Get relu layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetReLULayer(
    const LayerParameter& param) {
  ReLUParameter_Engine engine = param.relu_param().engine();
  if (engine == ReLUParameter_Engine_DEFAULT) {
    engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = ReLUParameter_Engine_CUDNN;
#endif
  }
  if (engine == ReLUParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new ReLULayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == ReLUParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new CuDNNReLULayer<Dtype, MItype, MOtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(ReLU, GetReLULayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(ReLU, GetReLULayer,
                       (double), (double), (double));

// Get sigmoid layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetSigmoidLayer(
    const LayerParameter& param) {
  SigmoidParameter_Engine engine = param.sigmoid_param().engine();
  if (engine == SigmoidParameter_Engine_DEFAULT) {
    engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SigmoidParameter_Engine_CUDNN;
#endif
  }
  if (engine == SigmoidParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new SigmoidLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SigmoidParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new CuDNNSigmoidLayer<Dtype, MItype, MOtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer,
                       (double), (double), (double));

// Get softmax layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetSoftmaxLayer(
    const LayerParameter& param) {
  SoftmaxParameter_Engine engine = param.softmax_param().engine();
  if (engine == SoftmaxParameter_Engine_DEFAULT) {
    engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = SoftmaxParameter_Engine_CUDNN;
#endif
  }
  if (engine == SoftmaxParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new SoftmaxLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == SoftmaxParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new CuDNNSoftmaxLayer<Dtype, MItype, MOtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer,
                       (double), (double), (double));

template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetCaffeSoftmaxLayer(
    const LayerParameter& param) {
  return shared_ptr<Layer<Dtype, MItype, MOtype> >(
      new SoftmaxLayer<Dtype, MItype, MOtype>(param));
}

REGISTER_LAYER_CREATOR(Softmax, GetCaffeSoftmaxLayer,
                       (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CREATOR(Softmax, GetCaffeSoftmaxLayer,
                       (uint8_t), (uint8_t), (uint8_t));
REGISTER_LAYER_CREATOR(Softmax, GetCaffeSoftmaxLayer,
                       (uint16_t), (uint16_t), (uint16_t));
REGISTER_LAYER_CREATOR(Softmax, GetCaffeSoftmaxLayer,
                       (uint32_t), (uint32_t), (uint32_t));
REGISTER_LAYER_CREATOR(Softmax, GetCaffeSoftmaxLayer,
                       (uint64_t), (uint64_t), (uint64_t));

// Get tanh layer according to engine.
template<typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetTanHLayer(
    const LayerParameter& param) {
  TanHParameter_Engine engine = param.tanh_param().engine();
  if (engine == TanHParameter_Engine_DEFAULT) {
    engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
    engine = TanHParameter_Engine_CUDNN;
#endif
  }
  if (engine == TanHParameter_Engine_CAFFE
      || Caffe::GetDevice(param.device(), true)->backend() == BACKEND_OPENCL) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new TanHLayer<Dtype, MItype, MOtype>(param));
#ifdef USE_CUDNN
  } else if (engine == TanHParameter_Engine_CUDNN) {
    return shared_ptr<Layer<Dtype, MItype, MOtype> >(
        new CuDNNTanHLayer<Dtype, MItype, MOtype>(param));
#endif
  } else {
    LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
    throw;  // Avoids missing return warning
  }
}

REGISTER_LAYER_CREATOR(TanH, GetTanHLayer,
                       (float), (float), (float));
REGISTER_LAYER_CREATOR(TanH, GetTanHLayer,
                       (double), (double), (double));

#ifdef WITH_PYTHON_LAYER
template <typename Dtype, typename MItype, typename MOtype>
shared_ptr<Layer<Dtype, MItype, MOtype> > GetPythonLayer(
    const LayerParameter& param) {
  Py_Initialize();
  try {
    bp::object module = bp::import(param.python_param().module().c_str());
    bp::object layer = module.attr(param.python_param().layer().c_str())(param);
    return bp::extract<shared_ptr<
        PythonLayer<Dtype, MItype, MOtype> > >(layer)();
  } catch (bp::error_already_set) {
    PyErr_Print();
    throw;
  }
}

REGISTER_LAYER_CREATOR(Python, GetPythonLayer, (float), (float), (float));
#endif

// Layers that use their constructor as their default creator should be
// registered in their corresponding cpp files. Do not register them here.
}  // namespace caffe
