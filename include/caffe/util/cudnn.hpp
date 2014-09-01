#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/proto/caffe.pb.h"

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " cuDNN error."; \
  } while (0)

namespace caffe {

// TODO(cudnn): check existence, add to CUDN_CHECK
// const char* cudnnGetErrorString(curandStatus_t error);
//
namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensor4dDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnCreateTensor4dDescriptor(desc));
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
      n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensor4dDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  createTensor4dDesc<Dtype>(desc, n, c, h, w,
      stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
  CUDNN_CHECK(cudnnSetFilterDescriptor(*desc, dataType<Dtype>::type,
      n, c, h, w));
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensor4dDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
  CUDNN_CHECK(cudnnSetConvolutionDescriptor(*conv, bottom, filter,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* conv,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int stride_h, int stride_w) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(conv));
  CUDNN_CHECK(cudnnSetPoolingDescriptor(*conv, *mode, h, w,
        stride_h, stride_w));
}

}  // namespace cudnn
}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
