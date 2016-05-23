#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

namespace caffe {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void createTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    std::vector<int> shape,
    std::vector<int> stride) {
  CHECK_EQ(shape.size(), stride.size())
      << "Dimensions of shape and stride don't match !";
  // fill shape with 1 to create tensors with at least 4 dimensions
  // to prevent CUDNN_STATUS_BAD_PARAM error in CUDNN v4
  // TODO(christian.payer@gmx.net): check CUDNN doc, probably fixed
  // in newer versions
  for (int i = shape.size(); i < 4; ++i) {
    shape.push_back(1);
    stride.push_back(1);
  }
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, dataType<Dtype>::type,
        shape.size(), shape.data(), stride.data()));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    std::vector<int> shape) {
  // set up stride
  std::vector<int> stride(shape.size(), 1);
  for (int i = stride.size() - 2; i >= 0; --i) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }
  setTensorNdDesc<Dtype>(desc, shape, stride);
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#endif
}

template <typename Dtype>
inline void createNdFilterDesc(cudnnFilterDescriptor_t* desc,
    std::vector<int> shape) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, shape.size(), shape.data()));
#else
  CUDNN_CHECK(cudnnSetFilterNdDescriptor(*desc, dataType<Dtype>::type,
      shape.size(), shape.data()));
#endif
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void setNdConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    std::vector<int> pad, std::vector<int> stride) {
  int nbDims;
  std::vector<int> shape(pad.size() + 2);
  cudnnDataType_t cudnn_type;
#if CUDNN_VERSION_MIN(5, 0, 0)
  cudnnTensorFormat_t tensor_format;
  cudnnGetFilterNdDescriptor(filter,
      shape.size(), &cudnn_type, &tensor_format, &nbDims, shape.data());
#else
  cudnnGetFilterNdDescriptor(filter,
      shape.size(), &cudnn_type, &nbDims, shape.data());
#endif
  CHECK_EQ(nbDims, pad.size() + 2)
      << "Dimensions of filters and pad don't match !";
  CHECK_EQ(nbDims, stride.size() + 2)
      << "Dimensions of filters and stride don't match !";
  std::vector<int> upscale(pad.size(), 1);
  CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(*conv,
      pad.size(), pad.data(), stride.data(), upscale.data(),
      CUDNN_CROSS_CORRELATION, cudnn_type));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode,
      CUDNN_PROPAGATE_NAN, h, w, pad_h, pad_w, stride_h, stride_w));
#else
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
      pad_h, pad_w, stride_h, stride_w));
#endif
}

template <typename Dtype>
inline void createActivationDescriptor(cudnnActivationDescriptor_t* activ_desc,
    cudnnActivationMode_t mode) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activ_desc, mode,
                                           CUDNN_PROPAGATE_NAN, Dtype(0)));
}

template <typename Dtype>
inline void createNdPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    std::vector<int> shape, std::vector<int> pad,
    std::vector<int> stride) {
  CHECK_EQ(shape.size(), pad.size())
      << "Dimensions of shape and pad don't match !";
  CHECK_EQ(shape.size(), stride.size())
      << "Dimensions of shape and stride don't match !";
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc, *mode,
      CUDNN_PROPAGATE_NAN, shape.size(), shape.data(), pad.data(),
      stride.data()));
#else
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc, *mode, shape.size(),
      shape.data(), pad.data(), stride.data()));
#endif
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
