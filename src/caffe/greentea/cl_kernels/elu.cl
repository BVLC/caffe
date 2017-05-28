#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(elu_forward,Dtype)(const int n, __global const Dtype* in,
                                          __global Dtype* out,
                                          Dtype alpha) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - 1.0);
  }
}

__kernel void TEMPLATE(elu_backward,Dtype)(const int n, __global const Dtype* in_diff,
                                           __global const Dtype* out_data,
                                           __global const Dtype* in_data,
                                           __global Dtype* out_diff,
                                           Dtype alpha) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] =
        in_data[index] > 0 ?
            in_diff[index] : in_diff[index] * (out_data[index] + alpha);
  }
}
