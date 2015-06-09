#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(concat,Dtype)(const int nthreads, __global const Dtype* in_data,
                                     const int forward, const int num_concats,
                                     const int concat_size,
                                     const int top_concat_axis,
                                     const int bottom_concat_axis,
                                     const int offset_concat_axis,
                                     __global Dtype* out_data) {

  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index
        + (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward == 1) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}
