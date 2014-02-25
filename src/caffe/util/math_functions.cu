// Copyright 2013 Yangqing Jia
// Copyright 2014 kloudkl@github

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <math_functions.h> // CUDA's, not caffe's, for fabs

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

template<typename Dtype>
__global__ void sign_kernel(const int n, const Dtype* x, Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = (Dtype(0) < x[index]) - (x[index] < Dtype(0));
  }
}

template <>
void caffe_gpu_sign<float>(const int n, const float* x, float* y) {
  sign_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

template <>
void caffe_gpu_sign<double>(const int n, const double* x, double* y) {
  sign_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

template<typename Dtype>
__global__ void fabs_kernel(const int n, const Dtype* x, Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = fabs(x[index]);
  }
}

template <>
void caffe_gpu_fabs<float>(const int n, const float* x, float* y) {
  fabs_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

template <>
void caffe_gpu_fabs<double>(const int n, const double* x, double* y) {
  fabs_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, x, y);
}

}  // namespace caffe
