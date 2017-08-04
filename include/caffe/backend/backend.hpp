#ifndef CAFFE_BACKEND_BACKEND_HPP_
#define CAFFE_BACKEND_BACKEND_HPP_

#include "caffe/backend/dev_ptr.hpp"

enum Backend {
  BACKEND_CUDA,
  BACKEND_OPENCL,
  BACKEND_HIP,
  BACKEND_CPU
};



#endif  // CAFFE_BACKEND_BACKEND_HPP_
