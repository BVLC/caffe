#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(im2col_sk,Dtype)(const int n,
                                        __global const Dtype* data_im,
                                        const int data_offset, const int height,
                                        const int width, const int kernel_h,
                                        const int kernel_w,
                                        const int ext_kernel_h,
                                        const int ext_kernel_w, const int pad_h,
                                        const int pad_w, const int stride_h,
                                        const int stride_w, const int kstride_h,
                                        const int kstride_w,
                                        const int height_col,
                                        const int width_col,
                                        __global Dtype* data_col) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    __global Dtype* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    __global const Dtype* data_im_ptr = data_im + data_offset;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ext_kernel_h; i += kstride_h) {
      for (int j = 0; j < ext_kernel_w; j += kstride_w) {
        int h = h_in + i;
        int w = w_in + j;
        (*data_col_ptr) =
            (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }

}

__kernel void TEMPLATE(col2im_sk,Dtype)(const int n, __global const Dtype* data_col,
                                        const int height, const int width,
                                        const int channels, const int patch_h,
                                        const int patch_w,
                                        const int ext_patch_h,
                                        const int ext_patch_w, const int pad_h,
                                        const int pad_w, const int stride_h,
                                        const int stride_w, const int kstride_h,
                                        const int kstride_w,
                                        const int height_col,
                                        const int width_col,
                                        __global Dtype* data_im, const int data_offset) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int width_col_1 = width_col - 1;
    int height_col_1 = height_col - 1;
    int w_col_start = (w < ext_patch_w) ? w % kstride_w : (w - ext_patch_w) + 1;
    int w_col_end =
        (w >= width_col) ?
            width_col_1 - (width_col_1 - w_col_start) % kstride_w : w;
    int h_col_start = (h < ext_patch_h) ? h % kstride_h : (h - ext_patch_h) + 1;
    int h_col_end =
        (h >= height_col) ?
            height_col_1 - (height_col_1 - h_col_start) % kstride_h : h;
    int w_num = (w - w_col_start) / kstride_w;
    int h_num = (h - h_col_start) / kstride_h;

    int coeff_w_idx = height_col * width_col;
    int coeff_h_idx = patch_w * coeff_w_idx;
    int offset = c * patch_h * coeff_h_idx;
    for (int h_col = h_col_start, h_idx = h_num; h_col <= h_col_end; h_col +=
        kstride_h, --h_idx) {
      for (int w_col = w_col_start, w_idx = w_num; w_col <= w_col_end; w_col +=
          kstride_w, --w_idx) {
        //int c_col = c * patch_h * patch_w + (h - h_col) / kstride_h * patch_w + (w - w_col) / kstride_w;
        //int c_col = c * patch_h * patch_w + h_idx * patch_w + w_idx;
        //val += data_col[(c_col * height_col + h_col) * width_col + w_col];
        val += data_col[offset + h_idx * coeff_h_idx + w_idx * coeff_w_idx
            + h_col * width_col + w_col];
      }
    }

    data_im[data_offset + index] = val;
  }
}
