
#if defined(_MSC_VER)

#include "caffe/common.hpp"

namespace caffe {

#if !defined(USE_CUDNN)

char GET_CLASS_GUARD_NAME(CuDNNConvolutionLayer);
char GET_CLASS_GUARD_NAME(CuDNNLCNLayer);
char GET_CLASS_GUARD_NAME(CuDNNLRNLayer);
char GET_CLASS_GUARD_NAME(CuDNNPoolingLayer);
char GET_CLASS_GUARD_NAME(CuDNNReLULayer);
char GET_CLASS_GUARD_NAME(CuDNNSigmoidLayer);
char GET_CLASS_GUARD_NAME(CuDNNSoftmaxLayer);
char GET_CLASS_GUARD_NAME(CuDNNTanHLayer);

#endif  // #if !defined(USE_CUDNN)

#if !defined(USE_OPENCV)
char GET_CLASS_GUARD_NAME(WindowDataLayer);

#endif  // #if !defined(USE_CUDNN)
}  // namespace caffe

#endif

