#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

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
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int_tp n, int_tp c, int_tp h, int_tp w,
    int_tp stride_n, int_tp stride_c, int_tp stride_h, int_tp stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int_tp n, int_tp c, int_tp h, int_tp w) {
  const int_tp stride_w = 1;
  const int_tp stride_h = w * stride_w;
  const int_tp stride_c = h * stride_h;
  const int_tp stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int_tp n, int_tp c, int_tp h, int_tp w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
      n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int_tp pad_h, int_tp pad_w, int_tp stride_h, int_tp stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int_tp h, int_tp w, int_tp pad_h, int_tp pad_w,
    int_tp stride_h, int_tp stride_w) {
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
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
