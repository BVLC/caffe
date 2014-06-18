// Copyright 2014 BVLC and contributors.

#include "caffe/common.hpp"
#include "caffe/util/opencl_device.hpp"

namespace caffe {


#define DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__kernel void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  OPENCL_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_opencl_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_opencl_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template<typename Dtype> \
void OpenCLDevice<Dtype>::name(const int N, const Dtype* x, Dtype* y) { \
  caffe_opencl_##name<Dtype>(N, x, y); \
} \
template \
void OpenCLDevice<float>::name(const int N, const float* x, float* y); \
template \
void OpenCLDevice<double>::name(const int N, const double* x, double* y);


#define DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(name, operation) \
template <typename Dtype> \
__kernel void name##_kernel(__global const int n, __global const Dtype* a, \
                            __global const Dtype* b, __global Dtype* y) { \
  OPENCL_KERNEL_LOOP(i, n) { \
    operation; \
  } \
} \
template <> \
void caffe_opencl_##name<float>( \
    __global const int N, __global const float* a, \
    __global const float* b, __global float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */  \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>( \
      N, a, b, y); \
} \
template <> \
void caffe_opencl_##name<double>( \
    __global const int N, __global const double* a, \
    __global const double* b, __global double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */  \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>( \
      N, a, b, y); \
} \
template<typename Dtype> \
void OpenCLDevice<Dtype>::name(const int N, const Dtype* a, const Dtype* b, \
                               Dtype* y) { \
  caffe_opencl_##name<Dtype>(N, x, y); \
} \
template \
void OpenCLDevice<float>::name(const int N, const float* a, const float* b, \
                               float* y); \
template \
void OpenCLDevice<double>::name(const int N, const double* a, \
                                const double* b, double* y);


DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sqr, y[i] = x[i] * x[i]);
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(exp, y[i] = exp(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sign, y[i] = sign<Dtype>(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(sgnbit, y[i] = signbit(x[i]));
DEFINE_AND_INSTANTIATE_OPENCL_UNARY_FUNC(fabs, y[i] = fabs(x[i]));

DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(add, y[i] = a[i] + b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(sub, y[i] = a[i] - b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(mul, y[i] = a[i] * b[i]);
DEFINE_AND_INSTANTIATE_OPENCL_BINARY_FUNC(div, y[i] = a[i] / b[i]);

}  // namespace caffe
