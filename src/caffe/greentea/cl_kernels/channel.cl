#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#define FLT_MAX 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void kernel_channel_max_s(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const float* data,
                                   __global float* out) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

__kernel void kernel_channel_max_d(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const double* data,
                                   __global double* out) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    double maxval = (double) -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

__kernel void kernel_channel_subtract_s(const int count, const int num,
                                        const int channels,
                                        const int spatial_dim,
                                        __global const float* channel_max,
                                        __global float* data) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__kernel void kernel_channel_subtract_d(const int count, const int num,
                                        const int channels,
                                        const int spatial_dim,
                                        __global const double* channel_max,
                                        __global double* data) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__kernel void kernel_exp_s(const int count, __global const float* data,
                           __global float* out) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    out[index] = exp(data[index]);
  }
}

__kernel void kernel_exp_d(const int count, __global const double* data,
                           __global double* out) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    out[index] = exp(data[index]);
  }
}

__kernel void kernel_channel_sum_s(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const float* data,
                                   __global float* channel_sum) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__kernel void kernel_channel_sum_d(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const double* data,
                                   __global double* channel_sum) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    double sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__kernel void kernel_channel_div_s(const int count, const int num,
                                   const int channels, const int spatial_dim,
                                   __global const float* channel_sum,
                                   __global float* data) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

__kernel void kernel_channel_div_d(const int count, const int num,
                                   const int channels, const int spatial_dim,
                                   __global const double* channel_sum,
                                   __global double* data) {
  for (int index = get_global_id(0); index < count;
      index += get_global_size(0)) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

__kernel void kernel_channel_dot_s(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const float* data_1,
                                   __global const float* data_2,
                                   __global float* channel_dot) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

__kernel void kernel_channel_dot_d(const int num, const int channels,
                                   const int spatial_dim,
                                   __global const double* data_1,
                                   __global const double* data_2,
                                   __global double* channel_dot) {
  for (int index = get_global_id(0); index < num * spatial_dim; index +=
      get_global_size(0)) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    double dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}
