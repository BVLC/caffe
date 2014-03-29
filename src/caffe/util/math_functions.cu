// Copyright 2014 BVLC and contributors.

#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit, exp(f)
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(fabs, y[index] = fabs(x[index]));


template <typename Dtype>
__global__ void sigmoid_kernel(const int n, const Dtype* x, Dtype* y);

template <>
__global__ void sigmoid_kernel<float>(const int n, const float* x, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 1.0 / (1 + expf(-x[index]));
  }
}

template <>
__global__ void sigmoid_kernel<double>(const int n, const double* x, double* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = 1.0 / (1 + exp(-x[index]));
  }
}

template <typename Dtype>
void caffe_gpu_sigmoid(const int n, const Dtype* x, Dtype* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sigmoid_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

template
void caffe_gpu_sigmoid<float>(const int n, const float* x, float* y);
template
void caffe_gpu_sigmoid<double>(const int n, const double* x, double* y);

}  // namespace caffe
