#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(softmax_loss_forward,Dtype)(
    int n, __global const Dtype* prob_data, __global const Dtype* label,
    __global Dtype* loss,
    const int num, const int dim, const int spatial_dim,
    const int has_ignore_label_, const int ignore_label_,
    __global Dtype* counts) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = (int) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(
          max((Dtype) (prob_data[n * dim + label_value * spatial_dim + s]),
              (Dtype) FLT_MIN));
      counts[index] = 1;
    }
  }
}

__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int nthreads,
                                                    __global const Dtype* top,
                                                    __global const Dtype* label,
                                                    __global Dtype* bottom_diff,
                                                    const int num,
                                                    const int dim,
                                                    const int spatial_dim,
                                                    const int has_ignore_label_,
                                                    const int ignore_label_,
                                                    __global Dtype* counts) {

  const int channels = dim / spatial_dim;

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int label_value = (int) (label[n * spatial_dim + s]);

      if (has_ignore_label_ == 1 && label_value == ignore_label_) {
        for (int c = 0; c < channels; ++c) {
          bottom_diff[n * dim + c * spatial_dim + s] = 0;
        }
        counts[index] = 0;
      } else {
        bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
        counts[index] = 1;
      }
    }
  }
}
