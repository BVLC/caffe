#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(softmax_loss_forward_gpu,Dtype)(int n, __global const Dtype* prob_data,
                                         __global const Dtype* label,
                                         __global Dtype* loss, const int num,
                                         const int dim, const int spatial_dim,
                                         const int has_ignore_label_,
                                         const int ignore_label_,
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
          max((Dtype)(prob_data[n * dim + label_value * spatial_dim + s]),
          (Dtype)FLT_MIN));
      counts[index] = 1;
    }
  }
}
