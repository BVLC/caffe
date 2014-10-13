// Copyright 2014 BVLC and contributors.

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/local_update.hpp"

namespace caffe {

template <typename Dtype>
void local_update1_cpu(const Dtype* data_A, const Dtype* data_B, Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  int total = filter_num * location_num * output_num;
  for (int index=0; index<total; index++) {
    int p = index % location_num;
    int n = (index / location_num);
    data_R[index] = Dtype(0);

    for (int q=0; q<output_num; q++) {
      data_R[index] += data_A[q*location_num+p] * data_B[(q*filter_num+n)*location_num+p];
    }
  }
}

// Explicit instantiation
template void local_update1_cpu<float>(const float* data_A, const float* data_B,
                                   float* data_R, const int filter_num,
                                   const int location_num, const int output_num);
template void local_update1_cpu<double>(const double* data_A, const double* data_B,
                                   double* data_R, const int filter_num,
                                   const int location_num, const int output_num);

template <typename Dtype>
void local_update2_cpu(const Dtype* data_A, const Dtype* data_B,
                       Dtype* data_R, const int filter_num,
                       const int location_num, const int output_num) {
  int total = filter_num * location_num ;
  for (int index=0; index<total; index++) {
    int p = index % location_num;
    int n = (index / location_num);
    for (int q=0; q<output_num; q++) {
      data_R[index] += data_A[q*location_num+p] * data_B[(q*filter_num+n)*location_num+p];
    }
  }
}

// Explicit instantiation
template void local_update2_cpu<float>(const float* data_A, const float* data_B,
                                   float* data_R, const int filter_num,
                                   const int location_num, const int output_num);
template void local_update2_cpu<double>(const double* data_A, const double* data_B,
                                   double* data_R, const int filter_num,
                                   const int location_num, const int output_num);

}  // namespace caffe
