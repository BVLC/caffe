#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(fill,Dtype)(const int_tp n, const KERNEL_ARG_DTYPE alpha, __global Dtype* x,
                                   const int_tp offx) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    x[index + offx] = alpha;
  }
}
