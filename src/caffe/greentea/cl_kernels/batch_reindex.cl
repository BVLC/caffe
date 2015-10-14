#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(br_forward,Dtype)(const int count, const int inner_dim,
                                         __global const Dtype* in,
                                         __global const Dtype* permut,
                                         __global Dtype* out) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / (inner_dim);
    int in_n = (int) (permut[n]);
    out[index] = in[in_n * (inner_dim) + index % (inner_dim)];
  }
}

__kernel void TEMPLATE(br_backward,Dtype)(const int count, const int inner_dim,
                                          __global const Dtype* in,
                                          __global const Dtype* top_indexes,
                                          __global const Dtype* begins,
                                          __global const Dtype* counts,
                                          __global Dtype* out) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / (inner_dim);
    out[index] = 0;
    int lower = (int) (begins[n]);
    int upper = lower + (int) (counts[n]);
    for (int i = lower; i < upper; ++i) {
      int in_n = (int) (top_indexes[i]);
      out[index] += in[in_n * (inner_dim) + index % (inner_dim)];
    }
  }
}
