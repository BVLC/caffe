// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

/* grid stride kernel */
template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n;
        i += blockDim.x + gridDim.x)
  {
     y[i] = a[i] - b[i];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a,
    const float* b, float* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_kernel<float><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a,
    const double* b, double* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_kernel<double><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

}  // namespace caffe
