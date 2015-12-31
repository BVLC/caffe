#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(softmax_loss_forward,Dtype)(
    int_tp n, __global const Dtype* prob_data, __global const Dtype* label,
    __global Dtype* loss,
    const int_tp num, const int_tp dim, const int_tp spatial_dim,
    const int has_ignore_label_, const int_tp ignore_label_,
    __global Dtype* counts) {

  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);
    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log((Dtype)(
          max((Dtype) (prob_data[n * dim + label_value * spatial_dim + s]),
              (Dtype) FLT_MIN)));
      counts[index] = 1;
    }
  }
}

__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int_tp nthreads,
                                                    __global const Dtype* top,
                                                    __global const Dtype* label,
                                                    __global Dtype* bottom_diff,
                                                    const int_tp num,
                                                    const int_tp dim,
                                                    const int_tp spatial_dim,
                                                    const int has_ignore_label_,
                                                    const int_tp ignore_label_,
                                                    __global Dtype* counts) {

  const int_tp channels = dim / spatial_dim;

  for (int_tp index = get_global_id(0); index < nthreads; index +=
      get_global_size(0)) {

    const int_tp n = index / spatial_dim;
    const int_tp s = index % spatial_dim;
    const int_tp label_value = (int_tp) (label[n * spatial_dim + s]);

    if (has_ignore_label_ == 1 && label_value == ignore_label_) {
      for (int_tp c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}
