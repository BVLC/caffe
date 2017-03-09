#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(relu_forward,Dtype)(const int_tp n,
                                           __global const Dtype* in,
                                           __global Dtype* out,
                                           Dtype negative_slope) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

__kernel void TEMPLATE(relu_backward,Dtype)(const int_tp n,
                                            __global const Dtype* in_diff,
                                            __global const Dtype* in_data,
                                            __global Dtype* out_diff,
                                            Dtype negative_slope) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index]
        * ((in_data[index] > 0?1.0:0.0) + (in_data[index] <= 0?1.0:0.0) * negative_slope);
  }
}

__kernel void TEMPLATE(tanh_forward,Dtype)(const int_tp n,
                                           __global const Dtype* in,
                                           __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = tanh(in[index]);
  }
}

__kernel void TEMPLATE(tanh_backward,Dtype)(const int_tp n,
                                            __global const Dtype* in_diff,
                                            __global const Dtype* out_data,
                                            __global Dtype* out_diff) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

__kernel void TEMPLATE(sigmoid_forward,Dtype)(const int_tp n,
                                              __global const Dtype* in,
                                              __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = 1.0 / (1.0 + exp(-in[index]));
  }
}

__kernel void TEMPLATE(sigmoid_backward,Dtype)(const int_tp n,
                                               __global const Dtype* in_diff,
                                               __global const Dtype* out_data,
                                               __global Dtype* out_diff) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}

__kernel void TEMPLATE(threshold,Dtype)(const int_tp n, const Dtype threshold,
                                        __global const Dtype* in,
                                        __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > threshold ? 1.0 : 0.0;
  }
}

__kernel void TEMPLATE(prelu_forward,Dtype)(const int_tp n, const int_tp channels,
                                            const int_tp dim,
                                            __global const Dtype* in,
                                            __global Dtype* out,
                                            __global const Dtype* slope_data,
                                            const int_tp div_factor) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    int_tp c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

__kernel void TEMPLATE(prelu_backward,Dtype)(const int_tp n, const int_tp channels,
                                             const int_tp dim,
                                             __global const Dtype* in_diff,
                                             __global const Dtype* in_data,
                                             __global Dtype* out_diff,
                                             __global const Dtype* slope_data,
                                             const int_tp div_factor) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    int_tp c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index]
        * ((Dtype)(in_data[index] > 0?1.0:0.0) + (Dtype)(in_data[index] <= 0?1.0:0.0) * slope_data[c]);
  }
}

__kernel void TEMPLATE(prelu_param_backward,Dtype)(const int_tp n, const int_tp rows,
                                                   const int_tp rowPitch,
                                                   __global const Dtype* in_diff,
                                                   __global const Dtype* in_data,
                                                   __global Dtype* out_diff) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0?1.0:0.0);
    for (int k = 1; k < rows; k++) {
      out_diff[index] += in_diff[index + k * rowPitch]
          * in_data[index + k * rowPitch]
          * (in_data[index + k * rowPitch] <= 0?1.0:0.0);
    }
  }
}

__kernel void TEMPLATE(sce_loss_forward,Dtype)(const int_tp nthreads,
                                        __global const Dtype* input_data,
                                        __global const Dtype* target,
                                        __global Dtype* loss,
                                        const int_tp has_ignore_label_,
                                        const int_tp ignore_label_,
                                        __global Dtype* counts) {
  for (int_tp i = get_global_id(0); i < nthreads; i += get_global_size(0)) {
    const int_tp target_value = (int_tp)(target[i]);
    if (has_ignore_label_ == 1 && target_value == ignore_label_) {
      loss[i] = 0.0;
      counts[i] = 0.0;
    } else {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0.0)) -
          log((Dtype)1.0 + exp(input_data[i] - (Dtype)2.0 * input_data[i] *
          (input_data[i] >= 0.0)));
      counts[i] = 1.0;
    }
  }
}

__kernel void TEMPLATE(sce_loss_ignore_diff,Dtype)(const int_tp count,
                                        const int_tp ignore_label,
                                        __global const Dtype* target,
                                        __global Dtype* diff) {
  for (int_tp i = get_global_id(0); i < count; i += get_global_size(0)) {
    const int_tp target_value = (int_tp)(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0.0;
    }
  }
}
