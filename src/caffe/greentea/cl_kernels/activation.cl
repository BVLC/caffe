#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(relu_forward,Dtype)(const int n, __global const Dtype* in,
                             __global Dtype* out, Dtype negative_slope) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}
