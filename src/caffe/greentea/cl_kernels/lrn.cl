#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int nthreads,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 const Dtype negative_beta,
                                                 __global Dtype* out) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int nthreads, __global const Dtype* in,
                             const int num, const int channels,
                             const int height, const int width, const int size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* scale_off = scale + offset;
    int head = 0;
    const int pre_pad = (size - 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_scale = 0;
    // fill the scale at [n, :, h, w]
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      ++head;
    }
  }
}

__kernel void TEMPLATE(lrn_compute_diff,Dtype)(const int nthreads,
                               __global const Dtype* bottom_data,
                               __global const Dtype* top_data,
                               __global const Dtype* scale,
                               __global const Dtype* top_diff, const int num,
                               const int channels, const int height,
                               const int width, const int size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int n = index / width / height;
    const int offset = (n * channels * height + h) * width + w;
    const int step = height * width;
    __global const Dtype* bottom_off = bottom_data + offset;
    __global const Dtype* top_off = top_data + offset;
    __global const Dtype* scale_off = scale + offset;
    __global Dtype* top_diff_off = top_diff + offset;
    __global Dtype* bottom_diff_off = bottom_diff + offset;
    int head = 0;
    const int pre_pad = size - (size + 1) / 2;
    const int post_pad = size - pre_pad - 1;
    Dtype accum_ratio = 0;
    // accumulate values
    while (head < post_pad && head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      ++head;
    }
    // both add and subtract
    while (head < channels) {
      accum_ratio += top_diff_off[head * step] * top_off[head * step]
          / scale_off[head * step];
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_ratio -= top_diff_off[(head - size) * step]
            * top_off[(head - size) * step] / scale_off[(head - size) * step];
      }
      bottom_diff_off[(head - post_pad) * step] = top_diff_off[(head - post_pad)
          * step] * pow(scale_off[(head - post_pad) * step], negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}
