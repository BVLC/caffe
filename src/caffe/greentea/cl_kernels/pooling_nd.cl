#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(max_pool_forward_nd, Dtype)(const int_tp n,
                                                   const int_tp num_axes,
                                                   __global const Dtype* bottom_data,
                                                   const int_tp channels,
                                                   __global const int_tp* size,
                                                   __global const int_tp* pooled_size,
                                                   __global const int_tp* kernel_size,
                                                   __global const int_tp* ext_kernel_size,
                                                   __global const int_tp* stride,
                                                   __global const int_tp* dilation,
                                                   __global const int_tp* pad,
                                                   __global Dtype* top_data,
                                                   const int use_mask,
                                                   __global int_tp* mask, __global Dtype* top_mask) {
  int_tp d_idx[6];
  int_tp d_start[6];
  int_tp d_end[6];
  int_tp d_iter[6];
  int_tp i;

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    int_tp offset = 1;
    int_tp num = index;

    bool do_continue = false;

    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % pooled_size[i];
      d_start[i] = d_idx[i] * stride[i] - pad[i];
      d_end[i] = min(d_start[i] + ext_kernel_size[i], size[i]);
      d_start[i] = max(d_start[i], (int_tp)0);
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
        do_continue = true;
      }
    }

    if(do_continue) {
      continue;
    }

    int_tp chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype maxval = -FLT_MAX;
    int_tp maxidx = -1;
    int_tp final_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      int_tp size_prod = 1;
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
        if (d_iter[i] >= d_end[i] - dilation[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += dilation[i];
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


__kernel void TEMPLATE(max_pool_backward_nd, Dtype)(const int_tp n,
                                                    const int_tp num_axes,
                                                    __global const Dtype* top_diff,
                                                    const int use_mask,
                                                    __global const int_tp* mask,
                                                    __global const Dtype* top_mask,
                                                    const int_tp channels,
                                                    __global const int_tp* size,
                                                    __global const int_tp* pooled_size,
                                                    __global const int_tp* kernel_size,
                                                    __global const int_tp* ext_kernel_size,
                                                    __global const int_tp* stride,
                                                    __global const int_tp* dilation,
                                                    __global const int_tp* pad,
                                                    __global Dtype* bottom_diff) {
  int_tp d_idx[6];
  int_tp d_start[6];
  int_tp d_end[6];
  int_tp d_iter[6];
  int_tp i;

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    // find out the local index
    // find out the local offset
    int_tp offset = 1;
    int_tp num = index;
    for (i = num_axes - 1; i >= 0; --i) {
      d_idx[i] = num % size[i];
      if (dilation[i] > 1) {
        d_start[i] =
            (d_idx[i] < ext_kernel_size[i]) ?
                d_idx[i] % dilation[i] : (d_idx[i] - ext_kernel_size[i]) + 1;
        d_end[i] =
            (d_idx[i] >= pooled_size[i]) ?
                (pooled_size[i] - 1)
                    - (pooled_size[i] - 1 - d_start[i]) % dilation[i] :
                d_idx[i];
      } else {
        d_start[i] =
            (d_idx[i] + pad[i] < kernel_size[i]) ?
                0 : (d_idx[i] + pad[i] - kernel_size[i]) / stride[i] + 1;
        d_end[i] = min((int_tp) ((d_idx[i] + pad[i]) / stride[i] + 1),
                       (int_tp) (pooled_size[i]));
      }
      num /= size[i];
      offset *= pooled_size[i];
      d_iter[i] = d_start[i];

      if (d_start[i] > d_end[i]) {
        bottom_diff[index] = 0;
        return;
      }
    }
    int_tp chan = num % channels;
    num /= channels;
    offset *= (num * channels + chan);

    Dtype gradient = 0;
    int_tp final_offset = 0;
    int_tp im_offset = 0;

    bool incremented;
    do {
      final_offset = offset;
      im_offset = 0;
      int_tp size_prod = 1;
      int_tp pooled_size_prod = 1;
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
        if (d_iter[i] > d_end[i] - dilation[i]) {
          d_iter[i] = d_start[i];
        } else {
          d_iter[i] += dilation[i];
          incremented = true;
          break;
        }
      }
    } while (incremented);
    bottom_diff[index] = gradient;
  }
}
