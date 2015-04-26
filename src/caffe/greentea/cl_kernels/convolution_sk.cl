#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void convolution_sk_s(__global const float *w, __global const float *in,
                          const int in_off, const int height,
                          const int width, const int kernel_h,
                          const int kernel_w, const int ext_kernel_h,
                          const int ext_kernel_w, const int stride_h,
                          const int stride_w, const int kstride_h,
                          const int kstride_w, __global float *out, const int out_off) {



  for (int index = get_global_id(0); index < 0; index += get_global_size(0)) {

//(*(out+))

  }
}
