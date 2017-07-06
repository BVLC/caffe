#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(lrn_compute_output,Dtype)(const int_tp nthreads,
                                                 __global const Dtype* in,
                                                 __global const Dtype* scale,
                                                 const KERNEL_ARG_DTYPE negative_beta,
                                                 __global Dtype* out) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    out[index] = in[index] * pow(scale[index], (Dtype)negative_beta);
  }
}

__kernel void TEMPLATE(lrn_fill_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,
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
                               const KERNEL_ARG_DTYPE negative_beta,
                               const KERNEL_ARG_DTYPE cache_ratio,
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
          * step] * pow(scale_off[(head - post_pad) * step], (Dtype)negative_beta)
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
          * step] * pow(scale_off[(head - post_pad) * step], (Dtype)negative_beta)
          - cache_ratio * bottom_off[(head - post_pad) * step] * accum_ratio;
      ++head;
    }
  }
}

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable

#define SIMD_WIDTH 16 
#define TILE_W SIMD_WIDTH
#define TILE_H 8

#ifndef BEIGNET
__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
#endif
// Fuse pooling max layer into LRN across channel layer.
// Currently, only support non-padding, non-dilation mode and pool_w/h == pool_stride_w + 1.
// This kernel only get better performance on those Intel platforms with edram.
__kernel void TEMPLATE(lrn_fuse_pool_max,Dtype)(
                             __global const Dtype* in,
                             const int_tp channels,
                             const int_tp height, const int_tp width,
                             const int_tp tiled_height, int_tp tiled_width,
                             const int_tp size,
                             const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,
                             __global Dtype* const out,
                             const KERNEL_ARG_DTYPE negative_beta,
                             const int_tp pool_h, const int_tp pool_w, const int_tp pool_stride_h, int_tp pool_stride_w,
                             const int_tp pooled_height, const int_tp pooled_width,
                             const int_tp tile_pooled_block_h, const int_tp tile_pooled_block_w) {
  // find out the local offset
  const int_tp block_x = get_global_id(0) % tiled_width;
  const int_tp block_y = (get_global_id(0) / tiled_width) % tiled_height;
  const int_tp n = get_global_id(0) / (tiled_width * tiled_height);
  
  const int_tp w = block_x * tile_pooled_block_w * pool_stride_w;
  const int_tp h = block_y * tile_pooled_block_h * pool_stride_h;
  const int_tp offset = (n * channels * height + h) * width + w;
  const int_tp out_h = block_y * tile_pooled_block_h;
  const int_tp out_w = block_x * tile_pooled_block_w;
  const int_tp out_offset = (n * channels * pooled_height + out_h) * pooled_width + out_w + get_local_id(1);
  const int_tp step = height * width;
  const int_tp out_step = pooled_height * pooled_width;
  __global const Dtype* in_off = in + offset + get_local_id(1);
  __global Dtype* out_off = out + out_offset;
  Dtype scale_val;
  int_tp head = 0;
  const int_tp pre_pad = (size - 1) / 2;
  const int_tp post_pad = size - pre_pad - 1;
  Dtype accum_scale[TILE_H] = {0};
  if (w + get_local_id(1) >= width)
    return;

  while ( head < channels + post_pad ) {
    int ph = 0;
    int cur_out_h = 0;
    Dtype output_val = -DTYPE_MAX;
    // fill the scale at [n, :, h, w]
    // accumulate values
    for( int lrn_out_h = 0; lrn_out_h < TILE_H && (lrn_out_h + h) < height; lrn_out_h++) {
      Dtype prev_val = accum_scale[lrn_out_h];
      // add
      if (head < channels) {
        prev_val += in_off[head * step + width * lrn_out_h] * in_off[head * step + width * lrn_out_h];
      }
      // subtract
      if (head - size >= 0) {
        prev_val -= in_off[(head - size) * step + width * lrn_out_h] * in_off[(head - size) * step + width * lrn_out_h];
      }
      // compute output.
      if (head >= post_pad) {
        scale_val = k + prev_val * alpha_over_size;
        Dtype tmp = -DTYPE_MAX;
        //if (w + get_local_id(1) < width)
          tmp = in_off[(head - post_pad) * step + width * lrn_out_h] * native_powr(scale_val, (Dtype)negative_beta);

        Dtype h_max_val = -DTYPE_MAX;
        int index = (get_local_id(1) * pool_stride_w) % SIMD_WIDTH;
        for(int i = 0; i < pool_w; i++) {
          Dtype val = intel_sub_group_shuffle(tmp, index);
          if (h_max_val < val && (index + w < width))
            h_max_val = val;

          index = (index + 1) % SIMD_WIDTH;
        }
        // update output value.
        output_val = (output_val > h_max_val) ?
                      output_val : h_max_val;
        // time to write previous output and move to next value
        if (lrn_out_h - cur_out_h + 1 == pool_h) {
          if (get_local_id(1) < tile_pooled_block_w && (out_w + get_local_id(1)) < pooled_width) {
            out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;
          
            output_val = h_max_val;
          }
          ++ph;
          cur_out_h += pool_stride_h;
        }
      }
      accum_scale[lrn_out_h] = prev_val;
    }
    // Handle the incomplete pool box
    // an incomplete tiling box and we are not hitting the end of the pooled output.
    if (head >= post_pad &&
        ph < tile_pooled_block_h &&
        ph + out_h < pooled_height &&
        get_local_id(1) < tile_pooled_block_w &&
        (out_w + get_local_id(1)) < pooled_width) {
      out_off[(head - post_pad) * out_step + ph * pooled_width] = output_val;
    }
    head++;
  }
}

#undef TILE_W
#undef TILE_H
#undef SIMD_WIDTH
#endif

__kernel void TEMPLATE(lrn_full_no_scale,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,
                             __global Dtype* const out,
                             const KERNEL_ARG_DTYPE negative_beta) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    Dtype scale_val;
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
      scale_val = k + accum_scale * alpha_over_size;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);
      ++head;
    }
  }
}

__kernel void TEMPLATE(lrn_full,Dtype)(const int_tp nthreads, __global const Dtype* in,
                             const int_tp num, const int_tp channels,
                             const int_tp height, const int_tp width, const int_tp size,
                             const KERNEL_ARG_DTYPE alpha_over_size, const KERNEL_ARG_DTYPE k,
                             __global Dtype* const scale,
                             __global Dtype* const out,
                             const KERNEL_ARG_DTYPE negative_beta) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    // find out the local offset
    const int_tp w = index % width;
    const int_tp h = (index / width) % height;
    const int_tp n = index / width / height;
    const int_tp offset = (n * channels * height + h) * width + w;
    const int_tp step = height * width;
    __global const Dtype* in_off = in + offset;
    __global Dtype* out_off = out + offset;
    __global Dtype* scale_off = scale + offset;
    Dtype scale_val;
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
      scale_val = k + accum_scale * alpha_over_size;
      scale_off[(head - post_pad) * step] = scale_val;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);
      ++head;
    }
    // subtract only
    while (head < channels + post_pad) {
      if (head - size >= 0) {
        accum_scale -= in_off[(head - size) * step]
            * in_off[(head - size) * step];
      }
      scale_val = k + accum_scale * alpha_over_size;
      scale_off[(head - post_pad) * step] = scale_val;
      out_off[(head - post_pad) * step] = in_off[(head - post_pad) * step] * (Dtype)native_powr((Dtype)scale_val, (Dtype)negative_beta);
      ++head;
    }
  }
}
