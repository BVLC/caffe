#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(crop_copy, Dtype)(const int_tp n,
                                         const int_tp height,
                                         const int_tp width,
                                         const int_tp src_inner_stride,
                                         const int_tp dest_inner_stride,
                                         __global const Dtype* src,
                                         const int_tp src_off,
                                         __global Dtype* dest,
                                         const int_tp dest_off) {
  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    int_tp src_start = index * src_inner_stride + src_off;
    int_tp dest_start = index * dest_inner_stride + dest_off;
    for (int_tp i = 0; i < width; ++i) {
      dest[dest_start + i] = src[src_start + i];
    }
  }
}
