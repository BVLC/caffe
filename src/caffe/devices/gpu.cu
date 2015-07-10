#ifndef CPU_ONLY

#include <cublas_v2.h>

#include "caffe/common.hpp"
#include "caffe/devices/gpu.hpp"

namespace caffe {

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template<>
void GPUDevice<float>::abs(const int n, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, a, y);
}

template<>
void GPUDevice<double>::abs(const int n, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, a, y);
}

}  // namespace caffe

#endif  // CPU_ONLY
