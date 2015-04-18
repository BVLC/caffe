#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#define FLT_MAX 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void relu_forward_s(const int n, __global const float* in,
                             __global float* out, float negative_slope) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

__kernel void relu_forward_d(const int n, __global const double* in,
                             __global double* out, double negative_slope) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
