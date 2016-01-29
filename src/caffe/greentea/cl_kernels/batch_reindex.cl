#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(br_forward,Dtype)(const int_tp count, const int_tp inner_dim,
                                         __global const Dtype* in,
                                         __global const Dtype* permut,
                                         __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int_tp n = index / (inner_dim);
    int_tp in_n = (int_tp) (permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

__kernel void TEMPLATE(br_backward,Dtype)(const int_tp count, const int_tp inner_dim,
                                          __global const Dtype* in,
                                          __global const Dtype* top_indexes,
                                          __global const Dtype* begins,
                                          __global const Dtype* counts,
                                          __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int_tp n = index / (inner_dim);
    out[index] = 0;
    int_tp lower = (int_tp) (begins[n]);
    int_tp upper = lower + (int_tp) (counts[n]);
    for (int_tp i = lower; i < upper; ++i) {
      int_tp in_n = (int_tp) (top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
}
