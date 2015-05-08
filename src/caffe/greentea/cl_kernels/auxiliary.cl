#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(gpu_set,Dtype)(const int n, const Dtype alpha, __global Dtype* y) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index] = alpha;
  }
}

