#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void kernel_mul_s(const int n, __global const float* a, const int offa,
                           __global float* b, const int offb, __global float* y,
                           const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] + b[index + offb];
  }
}

__kernel void kernel_mul_d(const int n, __global const double* a, const int offa,
                           __global double* b, const int offb, __global double* y,
                           const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] + b[index + offb];
  }
}
