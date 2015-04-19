#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#define FLT_MAX 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void max_pool_forward_sk_s(const int nthreads,
                                    __global float* bottom_data,
                                    const int num, const int channels,
                                    const int height, const int width,
                                    const int pooled_height,
                                    const int pooled_width, const int kernel_h,
                                    const int kernel_w, const int ext_kernel_h,
                                    const int ext_kernel_w, const int stride_h,
                                    const int stride_w, const int kstride_h,
                                    const int kstride_w, const int pad_h,
                                    const int pad_w, __global float* top_data,
                                    const int use_mask,
                                    __global int* mask,
                                    __global float* top_mask) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + ext_kernel_h, height);
    int wend = min(wstart + ext_kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    __global float* bottom_data_ptr = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        if (bottom_data_ptr[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data_ptr[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (use_mask == 1) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}


__kernel void max_pool_forward_sk_d(const int nthreads,
                                    __global const double* bottom_data,
                                    const int num, const int channels,
                                    const int height, const int width,
                                    const int pooled_height,
                                    const int pooled_width, const int kernel_h,
                                    const int kernel_w, const int ext_kernel_h,
                                    const int ext_kernel_w, const int stride_h,
                                    const int stride_w, const int kstride_h,
                                    const int kstride_w, const int pad_h,
                                    const int pad_w, __global double* top_data,
                                    __global int* mask,
                                    __global double* top_mask) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + ext_kernel_h, height);
    int wend = min(wstart + ext_kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    double maxval = -FLT_MAX;
    int maxidx = -1;
    __global float* bottom_data_ptr = bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += kstride_h) {
      for (int w = wstart; w < wend; w += kstride_w) {
        if (bottom_data_ptr[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data_ptr[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}
