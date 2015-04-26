#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void im2col_sk_gpu_kernel_s(const int n, __global const float* data_im,
                                     const int data_offset, const int height,
                                     const int width, const int kernel_h,
                                     const int kernel_w, const int ext_kernel_h,
                                     const int ext_kernel_w, const int pad_h,
                                     const int pad_w, const int stride_h,
                                     const int stride_w, const int kstride_h,
                                     const int kstride_w, const int height_col,
                                     const int width_col,
                                     __global float* data_col) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    __global float* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    __global const float* data_im_ptr = data_im + data_offset;
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

__kernel void im2col_sk_gpu_kernel_d(const int n,
                                     __global const double* data_im,
                                     const int data_offset, const int height,
                                     const int width, const int kernel_h,
                                     const int kernel_w, const int ext_kernel_h,
                                     const int ext_kernel_w, const int pad_h,
                                     const int pad_w, const int stride_h,
                                     const int stride_w, const int kstride_h,
                                     const int kstride_w, const int height_col,
                                     const int width_col,
                                     __global double* data_col) {

  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    __global double* data_col_ptr = data_col;
    data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
    __global const double* data_im_ptr = data_im + data_offset;
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
