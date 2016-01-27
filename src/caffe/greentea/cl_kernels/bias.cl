#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(bias_forward,Dtype)(const int_tp n,
                                           __global const Dtype* in,
                                           __global const Dtype* bias,
                                           const int_tp bias_dim,
                                           const int_tp inner_dim,
                                           __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

__kernel void TEMPLATE(scale_forward,Dtype)(const int_tp n,
                                            __global const Dtype* in,
                                            __global const Dtype* scale,
                                            const int_tp scale_dim,
                                            const int_tp inner_dim,
                                            __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}

__kernel void TEMPLATE(scale_bias_forward,Dtype)(const int_tp n,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 __global const Dtype* bias,
                                                 const int_tp scale_dim,
                                                 const int_tp inner_dim,
                                                 __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp scale_index = (index / inner_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}
