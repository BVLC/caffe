#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#define FLT_MIN 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void softmax_loss_forward_gpu_s(int n, __global const float* prob_data,
                                         __global const float* label,
                                         __global float* loss, const int num,
                                         const int dim, const int spatial_dim,
                                         const int has_ignore_label_,
                                         const int ignore_label_,
                                         __global float* counts) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max((float) (prob_data[n * dim + label_value * spatial_dim + s]),
              (float) FLT_MIN));
      counts[index] = 1;
    }
  }

}

__kernel void softmax_loss_forward_gpu_d(int n,
                                         __global const double* prob_data,
                                         __global const double* label,
                                         __global double* loss, const int num,
                                         const int dim, const int spatial_dim,
                                         const int has_ignore_label_,
                                         const int ignore_label_,
                                         __global double* counts) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max((double) (prob_data[n * dim + label_value * spatial_dim + s]),
              (double) FLT_MIN));
      counts[index] = 1;
    }
  }

}
