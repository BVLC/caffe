#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(relu_forward,Dtype)(const int n,
                                           __global const Dtype* in,
                                           __global Dtype* out,
                                           Dtype negative_slope) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

__kernel void TEMPLATE(relu_backward,Dtype)(const int n,
                                            __global const Dtype* in_diff,
                                            __global const Dtype* in_data,
                                            __global Dtype* out_diff,
                                            Dtype negative_slope) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index]
        * ((in_data[index] > 0) + (in_data[index] <= 0) * negative_slope);
  }
}

__kernel void TEMPLATE(tanh_forward,Dtype)(const int n,
                                           __global const Dtype* in,
                                           __global Dtype* out) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = tanh(in[index]);
  }
}

__kernel void TEMPLATE(tanh_backward,Dtype)(const int n,
                                            __global const Dtype* in_diff,
                                            __global const Dtype* out_data,
                                            __global Dtype* out_diff) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

__kernel void TEMPLATE(sigmoid_forward,Dtype)(const int n,
                                              __global const Dtype* in,
                                              __global Dtype* out) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}

__kernel void TEMPLATE(sigmoid_backward,Dtype)(const int n,
                                               __global const Dtype* in_diff,
                                               __global const Dtype* out_data,
                                               __global Dtype* out_diff) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

__kernel void TEMPLATE(threshold,Dtype)(const int n, const Dtype threshold,
                                        __global const Dtype* in,
                                        __global Dtype* out) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}

__kernel void TEMPLATE(prelu_forward,Dtype)(const int n, const int channels,
                                            const int dim,
                                            __global const Dtype* in,
                                            __global Dtype* out,
                                            __global const Dtype* slope_data,
                                            const int div_factor) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

__kernel void TEMPLATE(prelu_backward,Dtype)(const int n, const int channels,
                                             const int dim,
                                             __global const Dtype* in_diff,
                                             __global const Dtype* in_data,
                                             __global Dtype* out_diff,
                                             __global const Dtype* slope_data,
                                             const int div_factor) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index]
        * ((in_data[index] > 0) + (in_data[index] <= 0) * slope_data[c]);
  }
}

__kernel void TEMPLATE(prelu_param_backward,Dtype)(const int n,
                                                   __global const Dtype* in_diff, const int in_diff_off,
                                                   __global const Dtype* in_data, const int in_data_off,
                                                   __global Dtype* out_diff) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index + in_diff_off] * in_data[index + in_data_off] * (in_data[index + in_data_off] <= 0);
  }
}
