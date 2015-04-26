#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 0
#define FLT_MAX 0
#endif

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_fp64 : enable

__kernel void gpu_set_s(const int n, const float alpha, __global float* y) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index] = alpha;
  }
}

__kernel void gpu_set_d(const int n, const double alpha, __global double* y) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index] = alpha;
  }
}
