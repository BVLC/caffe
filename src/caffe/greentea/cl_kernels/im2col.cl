#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(im2col,Dtype)(const int_tp n,
                                        __global const Dtype* data_im,
                                        const int_tp data_offset, const int_tp height,
                                        const int_tp width, const int_tp kernel_h,
                                        const int_tp kernel_w,
                                        const int_tp ext_kernel_h,
                                        const int_tp ext_kernel_w, const int_tp pad_h,
                                        const int_tp pad_w, const int_tp stride_h,
                                        const int_tp stride_w, const int_tp dilation_h,
                                        const int_tp dilation_w,
                                        const int_tp height_col,
                                        const int_tp width_col,
                                        __global Dtype* data_col) {

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    int_tp w_out = index % width_col;
    int_tp h_index = index / width_col;
    int_tp h_out = h_index % height_col;
    int_tp channel_in = h_index / height_col;
    int_tp channel_out = channel_in * kernel_h * kernel_w;
    int_tp h_in = h_out * stride_h - pad_h;
    int_tp w_in = w_out * stride_w - pad_w;
    __global Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    __global const Dtype* data_im_ptr = data_im + data_offset;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int_tp i = 0; i < ext_kernel_h; i += dilation_h) {
      for (int_tp j = 0; j < ext_kernel_w; j += dilation_w) {
        int_tp h = h_in + i;
        int_tp w = w_in + j;
        (*data_col_ptr) =
            (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }

}

__kernel void TEMPLATE(col2im,Dtype)(const int_tp n,
                                        __global const Dtype* data_col,
                                        const int_tp height, const int_tp width,
                                        const int_tp channels, const int_tp patch_h,
                                        const int_tp patch_w,
                                        const int_tp ext_patch_h,
                                        const int_tp ext_patch_w, const int_tp pad_h,
                                        const int_tp pad_w, const int_tp stride_h,
                                        const int_tp stride_w, const int_tp dilation_h,
                                        const int_tp dilation_w,
                                        const int_tp height_col,
                                        const int_tp width_col,
                                        __global Dtype* data_im,
                                        const int_tp data_offset) {

  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype val = 0;
    int_tp w = index % width + pad_w;
    int_tp h = (index / width) % height + pad_h;
    int_tp c = index / (width * height);
    // compute the start and end of the output
    int_tp width_col_1 = width_col - 1;
    int_tp height_col_1 = height_col - 1;
    int_tp w_col_start = (w < ext_patch_w) ? w % dilation_w : (w - ext_patch_w) + 1;
    int_tp w_col_end =
        (w >= width_col) ?
            width_col_1 - (width_col_1 - w_col_start) % dilation_w : w;
    int_tp h_col_start = (h < ext_patch_h) ? h % dilation_h : (h - ext_patch_h) + 1;
    int_tp h_col_end =
        (h >= height_col) ?
            height_col_1 - (height_col_1 - h_col_start) % dilation_h : h;
    int_tp w_num = (w - w_col_start) / dilation_w;
    int_tp h_num = (h - h_col_start) / dilation_h;

    int_tp coeff_w_idx = height_col * width_col;
    int_tp coeff_h_idx = patch_w * coeff_w_idx;
    int_tp offset = c * patch_h * coeff_h_idx;
    for (int_tp h_col = h_col_start, h_idx = h_num; h_col <= h_col_end; h_col +=
        dilation_h, --h_idx) {
      for (int_tp w_col = w_col_start, w_idx = w_num; w_col <= w_col_end; w_col +=
          dilation_w, --w_idx) {
        //int_tp c_col = c * patch_h * patch_w + (h - h_col) / dilation_h * patch_w + (w - w_col) / dilation_w;
        //int_tp c_col = c * patch_h * patch_w + h_idx * patch_w + w_idx;
        //val += data_col[(c_col * height_col + h_col) * width_col + w_col];
        val += data_col[offset + h_idx * coeff_h_idx + w_idx * coeff_w_idx
            + h_col * width_col + w_col];
      }
    }

    data_im[data_offset + index] = val;
  }
}
