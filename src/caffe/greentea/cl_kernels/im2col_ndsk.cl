#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(im2col_ndsk, Dtype)(const int_tp n, const int_tp num_axes,
                                        __global const Dtype* data_im,
                                        const int_tp data_off,
                                        __global const int_tp* im_shape,
                                        __global const int_tp* col_shape,
                                        __global const int_tp* kernel_shape,
                                        __global const int_tp* pad,
                                        __global const int_tp* stride,
                                        __global const int_tp* kstride,
                                        __global Dtype* data_col,
                                        const int_tp data_col_off) {
  int_tp d_temp[6];
  int_tp d_iter[6];
  int_tp i;

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_in = index;
    int_tp channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % col_shape[i + 1];
      channel_in /= col_shape[i + 1];
      channel_out *= kernel_shape[i];
    }
    channel_out *= channel_in;
    int_tp data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * stride[i] - pad[i];
      channel_in *= im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= col_shape[i + 1];
      d_iter[i] = 0;
    }
    __global Dtype* data_col_ptr = data_col + data_col_off + channel_out;
    __global const Dtype* data_im_ptr = data_im + data_off + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int_tp d_iter_im = d_iter[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < im_shape[i + 1];
        if (!in_range) {
          break;
        }
      }

      // Write column data
      if (in_range) {
        int_tp data_im_offset = d_iter[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= im_shape[i + 1];
          data_im_offset += d_iter[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }

      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        // Old: const int_tp d_max = kernel_shape[i];
        // New (strided, limit is the external kernel size):
        const int_tp d_max = (kernel_shape[i] - 1) * kstride[i] + 1;
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          // Old: ++d_iter[i];
          // New (strided, increment by the stride each time):
          d_iter[i] += kstride[i];
          incremented = true;
          break;
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }
}

__kernel void TEMPLATE(col2im_ndsk, Dtype)(const int_tp n, const int_tp num_axes,
                                  __global const Dtype* data_col,
                                    const int_tp data_col_off,
                                  __global const int_tp* im_shape,
                                  __global const int_tp* col_shape,
                                  __global const int_tp* kernel_shape,
                                  __global const int_tp* pad,
                                  __global const int_tp* stride,
                                  __global const int_tp* kstride,
                                  __global Dtype* data_im,
                                  const int_tp data_off) {
  int_tp d_im[6];
  int_tp d_col_size[6];
  int_tp d_col_iter[6];
  int_tp d_col_start[6];
  int_tp d_col_end[6];
  int_tp d_ext_patch[6];
  int_tp d_idx[6];

  for (int_tp i = num_axes - 1; i >= 0; --i) {
    d_ext_patch[i] = (kernel_shape[i] - 1) * kstride[i] + 1;
    d_col_size[i] = (im_shape[i + 1] + 2 * pad[i] - d_ext_patch[i])
        / stride[i] + 1;
  }

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_im = index;
    // Calculate d_im (image dimensions).
    for (int_tp i = num_axes - 1; i >= 0; --i) {
      d_im[i] = channel_im % im_shape[i + 1] + pad[i];
      channel_im /= im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int_tp i = 0; i < num_axes; ++i) {
      // Old:
      /*d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_shape[i]) ?
          0 : (d_im[i] - kernel_shape[i]) / stride[i] + 1;
      d_col_end[i] = min(d_im[i] / stride[i] + 1, col_shape[i + 1]);*/
      // New:
      d_col_start[i] = (d_im[i] < d_ext_patch[i]) ?
          d_im[i] % kstride[i] : (d_im[i] - d_ext_patch[i]) + 1;
      d_col_iter[i] = d_col_start[i];
      d_idx[i] = (d_im[i] - d_col_start[i]) / kstride[i];
      d_col_end[i] = (d_im[i] >= d_col_size[i]) ?
          (d_col_size[i] - 1) - ((d_col_size[i] - 1) - d_col_start[i])
          % kstride[i] : d_im[i];
      if (d_col_start[i] > d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int_tp i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    do {
      // Compute the final offset.
      int_tp final_offset = 0;
      int_tp coeff_prod = 1;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        final_offset +=  d_col_iter[i] * coeff_prod;
        coeff_prod *= d_col_size[i];
      }
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        final_offset += d_idx[i] * coeff_prod;
        coeff_prod *= kernel_shape[i];
      }
      final_offset += channel_im * coeff_prod;
      val += data_col[final_offset];
      incremented = false;
      for (int_tp i = num_axes - 1; i >= 0; --i) {
        if (d_col_iter[i] > d_col_end[i] - kstride[i]) {
          d_col_iter[i] = d_col_start[i];
          d_idx[i] = (d_im[i] - d_col_start[i]) / kstride[i];
        } else {  // d_col_iter[i] <= d_max - kstride[1]
          d_col_iter[i] += kstride[i];
          --d_idx[i];
          incremented = true;
          break;  // for (int_tp i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }
}
