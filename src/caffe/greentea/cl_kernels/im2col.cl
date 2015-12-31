#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(im2col,Dtype)(const int_tp n,
                                     __global const Dtype* data_im,
                                     const int_tp data_im_off,
                                     const int_tp height, const int_tp width,
                                     const int_tp kernel_h,
                                     const int_tp kernel_w, const int_tp pad_h,
                                     const int_tp pad_w, const int_tp stride_h,
                                     const int_tp stride_w,
                                     const int_tp dilation_h,
                                     const int_tp dilation_w,
                                     const int_tp height_col,
                                     const int_tp width_col,
                                     __global Dtype* data_col,
                                     const int_tp data_col_off) {

  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    const int_tp h_index = index / width_col;
    const int_tp h_col = h_index % height_col;
    const int_tp w_col = index % width_col;
    const int_tp c_im = h_index / height_col;
    const int_tp c_col = c_im * kernel_h * kernel_w;
    const int_tp h_offset = h_col * stride_h - pad_h;
    const int_tp w_offset = w_col * stride_w - pad_w;
    __global Dtype* data_col_ptr = data_col + data_col_off;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    __global const Dtype* data_im_ptr = data_im + data_im_off;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int_tp i = 0; i < kernel_h; ++i) {
      for (int_tp j = 0; j < kernel_w; ++j) {
        int_tp h_im = h_offset + i * dilation_h;
        int_tp w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
                data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

__kernel void TEMPLATE(col2im,Dtype)(const int_tp n,
                                     __global const Dtype* data_col,
                                     const int_tp data_col_off,
                                     const int_tp height, const int_tp width,
                                     const int_tp channels,
                                     const int_tp kernel_h,
                                     const int_tp kernel_w, const int_tp pad_h,
                                     const int_tp pad_w, const int_tp stride_h,
                                     const int_tp stride_w,
                                     const int_tp dilation_h,
                                     const int_tp dilation_w,
                                     const int_tp height_col,
                                     const int_tp width_col,
                                     __global Dtype* data_im,
                                     const int_tp data_im_off) {

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype val = 0;
    const int_tp w_im = index % width + pad_w;
    const int_tp h_im = (index / width) % height + pad_h;
    const int_tp c_im = index / (width * height);
    int_tp kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int_tp kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int_tp w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int_tp w_col_end = min(w_im / stride_w + 1, width_col);
    const int_tp h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int_tp h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int_tp h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int_tp w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int_tp h_k = (h_im - h_col * stride_h);
        int_tp w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int_tp data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_off + data_col_index];
        }
      }
    }
    data_im[data_im_off + index] = val;
  }
}
