#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(kernel_channel_max,Dtype)(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   __global const Dtype* data,
                                   __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int_tp c = 0; c < channels; ++c) {
      maxval = max((Dtype)(data[(n * channels + c) * spatial_dim + s]), (Dtype)maxval);
    }
    out[index] = maxval;
  }
}

__kernel void TEMPLATE(kernel_channel_subtract,Dtype)(const int_tp count, const int_tp num,
                                        const int_tp channels,
                                        const int_tp spatial_dim,
                                        __global const Dtype* channel_max,
                                        __global Dtype* data) {
  for (int_tp index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int_tp n = index / channels / spatial_dim;
    int_tp s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__kernel void TEMPLATE(kernel_exp,Dtype)(const int_tp count, __global const Dtype* data,
                           __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    out[index] = exp(data[index]);
  }
}

__kernel void TEMPLATE(kernel_channel_sum,Dtype)(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   __global const Dtype* data,
                                   __global Dtype* channel_sum) {
  for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    Dtype sum = 0;
    for (int_tp c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__kernel void TEMPLATE(kernel_channel_div,Dtype)(const int_tp count, const int_tp num,
                                   const int_tp channels, const int_tp spatial_dim,
                                   __global const Dtype* channel_sum,
                                   __global Dtype* data) {
  for (int_tp index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int_tp n = index / channels / spatial_dim;
    int_tp s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

__kernel void TEMPLATE(kernel_channel_dot,Dtype)(const int_tp num, const int_tp channels,
                                   const int_tp spatial_dim,
                                   __global const Dtype* data_1,
                                   __global const Dtype* data_2,
                                   __global Dtype* channel_dot) {
  for (int_tp index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int_tp n = index / spatial_dim;
    int_tp s = index % spatial_dim;
    Dtype dot = 0;
    for (int_tp c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}
