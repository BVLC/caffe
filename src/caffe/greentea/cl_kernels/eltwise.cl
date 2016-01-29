#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(eltwise_max_forward,Dtype)(
    const int_tp nthreads, __global const Dtype* bottom_data_a,
    __global const Dtype* bottom_data_b, const int_tp blob_idx,
    __global Dtype* top_data,
    __global int_tp* mask) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}

__kernel void TEMPLATE(eltwise_max_backward,Dtype)(const int_tp nthreads,
                                                   __global const Dtype* top_diff,
                                                   const int_tp blob_idx,
                                                   __global const int_tp* mask,
                                                   __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    Dtype gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}

