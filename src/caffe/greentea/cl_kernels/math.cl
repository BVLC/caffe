#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(kernel_mul,Dtype)(const int n, __global const Dtype* a, const int offa,
                           __global Dtype* b, const int offb, __global Dtype* y,
                           const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] + b[index + offb];
  }
}

