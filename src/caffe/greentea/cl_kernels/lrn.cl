#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int_tp nthreads,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 const Dtype negative_beta,
                                                 __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    out[index] = in[index] * pow(scale[index], negative_beta);
  }
}

__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const Dtype alpha_over_size, const Dtype k,
                             __global Dtype* const scale) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* scale_off = scale + offset;
    int_tp head = 0;
    const int_tp pre_pad = (size - 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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

__kernel void TEMPLATE(lrn_compute_diff,Dtype)(const int_tp nthreads,
                               __global const Dtype* bottom_data,
                               __global const Dtype* top_data,
                               __global const Dtype* scale,
                               __global const Dtype* top_diff, const int_tp num,
                               const int_tp channels, const int_tp height,
                               const int_tp width, const int_tp size,
                               const Dtype negative_beta,
                               const Dtype cache_ratio,
                               __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* bottom_off = bottom_data + offset;
    __global const Dtype* top_off = top_data + offset;
    __global const Dtype* scale_off = scale + offset;
    __global const Dtype* top_diff_off = top_diff + offset;
    __global Dtype* bottom_diff_off = bottom_diff + offset;
    int_tp head = 0;
    const int_tp pre_pad = size - (size + 1) / 2;
    const int_tp post_pad = size - pre_pad - 1;
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
