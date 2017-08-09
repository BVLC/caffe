#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void null_kernel(float arg) {
  float out = arg;
}

__kernel void fillbuffer(const int n, const char alpha, __global char* x,
                         const int offx) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    x[index + offx] = alpha;
  }
}


