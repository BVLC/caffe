#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(max_pool_forward_nd, Dtype)(const int n,
                                                   const int num_axes,
                                                   const __global Dtype* bottom_data,
                                                   const int channels,
                                                   __global const int* size,
                                                   __global const int* pooled_size,
                                                   __global const int* kernel_size,
                                                   __global const int* ext_kernel_size,
                                                   __global const int* stride,
                                                   __global const int* kstride,
                                                   __global const int* pad,
                                                   __global Dtype* top_data,
                                                   const int use_mask,
                                                   __global int* mask, __global Dtype* top_mask) {
  int d_idx[6];
  int d_start[6];
  int d_end[6];
  int d_iter[6];
  int i;

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int offset = 1;
    int num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = index % pooled_size[i];
      d_start[i] = d_idx[i] * stride[i] - pad[i];
      d_end[i] = min(d_start[i] + ext_kernel_size[i], size[i]);
      d_start[i] = max(d_start[i], 0);
      num /= pooled_size[i];
      offset *= size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] >= d_end[i]) {
        top_data[index] = -FLT_MAX;
        if (use_mask) {
          mask[index] = -1;
        } else {
          top_mask[index] = -1;
        }
        return;
      }
    }
    int chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    int final_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      int size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * size_prod;
        size_prod *= size[i];
      }

      if (bottom_data[final_offset] > maxval) {
        maxidx = final_offset;
        maxval = bottom_data[maxidx];
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] >= d_end[i] - kstride[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);

    top_data[index] = maxval;
    if (use_mask == 1) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}


__kernel void TEMPLATE(max_pool_backward_nd, Dtype)(const int n,
                                                    const int num_axes,
                                                    const __global Dtype* top_diff,
                                                    const int use_mask,
                                                    __global const int* mask,
                                                    __global const Dtype* top_mask,
                                                    const int channels,
                                                    __global const int* size,
                                                    __global const int* pooled_size,
                                                    __global const int* kernel_size,
                                                    __global const int* ext_kernel_size,
                                                    __global const int* stride,
                                                    __global const int* kstride,
                                                    __global const int* pad,
                                                    __global Dtype* bottom_diff) {
  int d_idx[6];
  int d_start[6];
  int d_end[6];
  int d_iter[6];
  int i;

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    int offset = 1;
    int num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % size[i];
      d_start[i] = (d_idx[i] < ext_kernel_size[i]) ?
          d_idx[i] % kstride[i] : (d_idx[i] - ext_kernel_size[i]) + 1;
      d_end[i] = (d_idx[i] >= pooled_size[i]) ?
          (pooled_size[i] - 1) - (pooled_size[i] - 1 - d_start[i]) %
          kstride[i] : d_idx[i];
      num /= size[i];
      offset *= pooled_size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] > d_end[i]) {
        bottom_diff[index] = 0;
        return;
      }
    }
    int chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype gradient = 0;
    int final_offset = 0;
    int im_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      im_offset = 0;
      int size_prod = 1;
      int pooled_size_prod = 1;
      for (i = num_axes - 1; i >= 0; --i) {
        final_offset += d_iter[i] * pooled_size_prod;
        im_offset += d_idx[i] * size_prod;
        size_prod *= size[i];
        pooled_size_prod *= pooled_size[i];
      }

      if (use_mask) {
        if (mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      } else {
        if (top_mask[final_offset] == im_offset) {
          gradient += top_diff[final_offset];
        }
      }

      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        if (d_iter[i] > d_end[i] - kstride[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);
    bottom_diff[index] = gradient;
  }
}
