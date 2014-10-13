// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/local_update.hpp"

namespace caffe {

template <typename Dtype>
__global__ void local_update1_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                    Dtype* data_R, const int filter_num,
                                    const int location_num, const int output_num) {
  int total = filter_num * location_num * output_num;
  CUDA_KERNEL_LOOP(index, total) {
    int p = index % location_num;
    int n = (index / location_num) % filter_num;
    int q = (index / location_num) / filter_num;
    data_R[index] += data_A[q*location_num+p] * data_B[n*location_num+p];
  }
}

template <typename Dtype>
void local_update1_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is filter_num x location_num
  // data_R is output_num x filter_num x location_num, the update performed is Rqnp += Aqp * Bnp

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update1_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(filter_num * location_num * output_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, filter_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update1_gpu<float>(const float* data_A, const float* data_B,
                                float* data_R, const int filter_num,
                                const int location_num, const int output_num);
template void local_update1_gpu<double>(const double* data_A, const double* data_B,
                                double* data_R, const int filter_num,
                                const int location_num, const int output_num);


template <typename Dtype>
__global__ void local_update2_gpu_kernel(const Dtype* data_A, const Dtype* data_B,
                                Dtype* data_R, const int filter_num,
                                const int location_num, const int output_num) {
  int total = filter_num * location_num;
  CUDA_KERNEL_LOOP(index, total) {
    int p = index % location_num;
    int n = (index / location_num);
    for (int q=0; q<output_num; q++) {
      data_R[index] += data_A[q*location_num+p] * data_B[(q*filter_num+n)*location_num+p];
    }
  }
}

template <typename Dtype>
void local_update2_gpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  // data_A is output_num x location_num
  // data_B is output_num x filter_num x location_num
  // data_R is filter_num x location_num, the update performed is Rnp += \sum_q(Aqp * Bqnp)

  // NOLINT_NEXT_LINE(whitespace/operators)
  local_update2_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(filter_num * location_num),
                             CAFFE_CUDA_NUM_THREADS>>>(data_A, data_B, data_R, filter_num, location_num, output_num);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void local_update2_gpu<float>(const float* data_A, const float* data_B,
                       float* data_R, const int filter_num,
                       const int location_num, const int output_num);
template void local_update2_gpu<double>(const double* data_A, const double* data_B,
                       double* data_R, const int filter_num,
                       const int location_num, const int output_num);

}  // namespace caffe
