#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(dropout_forward,Dtype)(const int n,
                                              __global const Dtype* in,
                                              __global const unsigned int* mask,
                                              const unsigned int threshold,
                                              const Dtype scale,
                                              __global Dtype* out) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

__kernel void TEMPLATE(dropout_backward,Dtype)(
    const int n, __global const Dtype* in_diff,
    __global const unsigned int* mask, const unsigned int threshold,
    const Dtype scale,
    __global Dtype* out_diff) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}
