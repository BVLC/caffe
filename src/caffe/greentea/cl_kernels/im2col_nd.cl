#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(im2col_nd, Dtype)(const int_tp n, const int_tp num_axes,
                                         const int_tp channel_axis,
                                         __global const Dtype* data_im,
                                         const int_tp data_im_off,
                                         __global const int_tp* im_shape,
                                         __global const int_tp* col_shape,
                                         __global const int_tp* kernel_shape,
                                         __global const int_tp* pad,
                                         __global const int_tp* stride,
                                         __global const int_tp* dilation,
                                         __global Dtype* data_col,
                                         const int_tp data_col_off) {
  int_tp d_temp[6];
  int_tp d_iter[6];
  int_tp i;

  __global const int_tp* im_shape_ptr = im_shape + channel_axis;
  __global const int_tp* col_shape_ptr = col_shape + channel_axis;

  __local int_tp shared_dilation[6];
  __local int_tp shared_kernel_shape[6];
  __local int_tp shared_pad[6];
  __local int_tp shared_stride[6];
  __local int_tp shared_col_shape[6 + 1];
  __local int_tp shared_im_shape[6 + 1];

  for (int li = get_local_id(0); li < num_axes; li += get_local_size(0)) {
    shared_dilation[li] = dilation[li];
    shared_kernel_shape[li] = kernel_shape[li];
    shared_pad[li] = pad[li];
    shared_stride[li] = stride[li];
  }

  for (int li = get_local_id(0); li < num_axes + 1; li += get_local_size(0)) {
    shared_col_shape[li] = col_shape_ptr[li];
    shared_im_shape[li] = im_shape_ptr[li];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp channel_in = index;
    int_tp channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int_tp data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    __global Dtype* data_col_ptr = data_col + data_col_off + channel_out;
    __global const Dtype* data_im_ptr = data_im + data_im_off + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int_tp d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) {
          break;
        }
      }
      if (in_range) {
        int_tp data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int_tp d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int_tp i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }
}

__kernel void TEMPLATE(col2im_nd, Dtype)(const int_tp n, const int_tp num_axes,
                                         const int_tp channel_axis,
                                         __global const Dtype* data_col,
                                         const int_tp data_col_off,
                                         __global const int_tp* im_shape,
                                         __global const int_tp* col_shape,
                                         __global const int_tp* kernel_shape,
                                         __global const int_tp* pad,
                                         __global const int_tp* stride,
                                         __global const int_tp* dilation,
                                         __global Dtype* data_im,
                                         const int_tp data_im_off) {
  int_tp d_im[6];
  int_tp d_col_iter[6];
  int_tp d_col_start[6];
  int_tp d_col_end[6];

  __global const int_tp* im_shape_ptr = im_shape + channel_axis;
  __global const int_tp* col_shape_ptr = col_shape + channel_axis;

  __local int_tp shared_dilation[6];
  __local int_tp shared_kernel_shape[6];
  __local int_tp shared_pad[6];
  __local int_tp shared_stride[6];
  __local int_tp shared_col_shape[6 + 1];
  __local int_tp shared_im_shape[6 + 1];

  for (int li = get_local_id(0); li < num_axes; li += get_local_size(0)) {
    shared_dilation[li] = dilation[li];
    shared_kernel_shape[li] = kernel_shape[li];
    shared_pad[li] = pad[li];
    shared_stride[li] = stride[li];
  }
  for (int li = get_local_id(0); li < num_axes + 1; li += get_local_size(0)) {
    shared_col_shape[li] = col_shape_ptr[li];
    shared_im_shape[li] = im_shape_ptr[li];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int_tp c_im = index;
    // Calculate d_im (image dimensions).
    for (int_tp i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int_tp i = 0; i < num_axes; ++i) {
      const int_tp kernel_extent = shared_dilation[i]
          * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ?
              0 : (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] = min(d_im[i] / shared_stride[i] + 1,
                         shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int_tp i = 0; i < num_axes; ++i)
      }
    }
    if (!done) {
      // Loop over the col to compute the output val.
      Dtype val = 0;
      bool incremented = true;
      bool skip = false;
      do {
        // Compute the final offset.
        int_tp final_offset = 0;
        int_tp kernel_shape_prod = 1;
        int_tp kernel_index;
        for (int_tp i = num_axes - 1; i >= 0; --i) {
          kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
          if (kernel_index % shared_dilation[i]) {
            skip = true;
            break;
          } else {
            kernel_index /= shared_dilation[i];
            final_offset += kernel_index * kernel_shape_prod;
            kernel_shape_prod *= shared_kernel_shape[i];
          }
        }
        if (!skip) {
          final_offset += kernel_shape_prod * c_im;
          for (int_tp i = 0; i < num_axes; ++i) {
            final_offset *= shared_col_shape[i + 1];
            final_offset += d_col_iter[i];
          }
          val += data_col[data_col_off + final_offset];
        }
        skip = false;
        incremented = false;
        for (int_tp i = num_axes - 1; i >= 0; --i) {
          const int_tp d_max = d_col_end[i];
          if (d_col_iter[i] == d_max - 1) {
            d_col_iter[i] = d_col_start[i];
          } else {  // d_col_iter[i] < d_max - 1
            ++d_col_iter[i];
            incremented = true;
            break;  // for (int_tp i = num_axes - 1; i >= 0; --i)
          }
        }  // for (int_tp i = num_axes - 1; i >= 0; --i)
      } while (incremented);
      data_im[data_im_off + index] = val;
    }
  }
}
