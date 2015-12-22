#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

<<<<<<< HEAD
#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

=======
>>>>>>> device-abstraction
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  static float oneval, zeroval;
  static const void *one, *zero;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  static float oneval, zeroval;
  static const void *one, *zero;
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/device/blob.hpp
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/caffe-merge
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/device/blob.hpp
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/device/blob.hpp
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/device/blob.hpp
=======
  static float oneval, zeroval;
  static const void *one, *zero;
=======
<<<<<<< HEAD
<<<<<<< HEAD
  static float oneval, zeroval;
  static const void *one, *zero;
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> origin/BVLC/parallel
=======
  static float oneval, zeroval;
  static const void *one, *zero;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  static float oneval, zeroval;
  static const void *one, *zero;
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  static float oneval, zeroval;
  static const void *one, *zero;
>>>>>>> device-abstraction
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> caffe
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
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
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
>>>>>>> pod/caffe-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
=======
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensor4dDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensor4dDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensor4dDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensor4dDescriptor_t* desc,
>>>>>>> origin/BVLC/parallel
=======
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> device-abstraction
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
                         stride_n, stride_c, stride_h, stride_w);
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
                         stride_n, stride_c, stride_h, stride_w);
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod/device/blob.hpp
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod/caffe-merge
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
<<<<<<< HEAD
<<<<<<< HEAD
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod/device/blob.hpp
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod-caffe-pod.hpp-merge
      stride_n, stride_c, stride_h, stride_w);
>>>>>>> origin/BVLC/parallel
=======
                         stride_n, stride_c, stride_h, stride_w);
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
                         stride_n, stride_c, stride_h, stride_w);
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod/device/blob.hpp
=======
                         stride_n, stride_c, stride_h, stride_w);
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
                         stride_n, stride_c, stride_h, stride_w);
=======
      stride_n, stride_c, stride_h, stride_w);
>>>>>>> origin/BVLC/parallel
=======
                         stride_n, stride_c, stride_h, stride_w);
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
                         stride_n, stride_c, stride_h, stride_w);
>>>>>>> device-abstraction
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod/caffe-merge
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod-caffe-pod.hpp-merge
  CUDNN_CHECK(cudnnSetFilterDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> origin/BVLC/parallel
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
<<<<<<< HEAD
<<<<<<< HEAD
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
<<<<<<< HEAD
<<<<<<< HEAD
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
<<<<<<< HEAD
<<<<<<< HEAD
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnSetFilterDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> origin/BVLC/parallel
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
>>>>>>> device-abstraction
      n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
    cudnnTensor4dDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolutionDescriptor(*conv, bottom, filter,
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
    cudnnTensor4dDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolutionDescriptor(*conv, bottom, filter,
>>>>>>> origin/BVLC/parallel
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> caffe
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
<<<<<<< HEAD
<<<<<<< HEAD
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
    cudnnTensor4dDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolutionDescriptor(*conv, bottom, filter,
>>>>>>> origin/BVLC/parallel
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
=======
    cudnnTensor4dDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolutionDescriptor(*conv, bottom, filter,
>>>>>>> origin/BVLC/parallel
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> caffe
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
>>>>>>> device-abstraction
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
inline void createPoolingDesc(cudnnPoolingDescriptor_t* conv,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int stride_h, int stride_w) {
>>>>>>> origin/BVLC/parallel
=======
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
=======
>>>>>>> pod/device/blob.hpp
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
=======
<<<<<<< HEAD
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
<<<<<<< HEAD
<<<<<<< HEAD
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
=======
inline void createPoolingDesc(cudnnPoolingDescriptor_t* conv,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int stride_h, int stride_w) {
>>>>>>> origin/BVLC/parallel
=======
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {
>>>>>>> device-abstraction
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
>>>>>>> pod/device/blob.hpp
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
>>>>>>> pod/caffe-merge
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
>>>>>>> pod/device/blob.hpp
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
>>>>>>> pod-caffe-pod.hpp-merge
    *mode = CUDNN_POOLING_AVERAGE;
>>>>>>> origin/BVLC/parallel
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
<<<<<<< HEAD
<<<<<<< HEAD
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> pod/device/blob.hpp
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> pod/device/blob.hpp
=======
    *mode = CUDNN_POOLING_AVERAGE;
>>>>>>> origin/BVLC/parallel
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
>>>>>>> device-abstraction
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(conv));
  CUDNN_CHECK(cudnnSetPoolingDescriptor(*conv, *mode, h, w,
        stride_h, stride_w));
>>>>>>> origin/BVLC/parallel
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
=======
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));
  CUDNN_CHECK(cudnnSetPooling2dDescriptor(*pool_desc, *mode, h, w,
        pad_h, pad_w, stride_h, stride_w));
>>>>>>> device-abstraction
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
