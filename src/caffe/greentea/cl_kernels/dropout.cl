#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(dropout_forward,Dtype)(const int_tp n,
                                              __global const Dtype* in,
                                              __global const uint_tp* mask,
                                              const uint_tp threshold,
                                              const Dtype scale,
                                              __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] * ((mask[index] > threshold)?1.0:0.0) * scale;
  }
}

__kernel void TEMPLATE(dropout_backward,Dtype)(
    const int_tp n, __global const Dtype* in_diff,
    __global const uint_tp* mask, const uint_tp threshold,
    const Dtype scale,
    __global Dtype* out_diff) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index] * ((mask[index] > threshold)?1.0:0.0) * scale;
  }
}
