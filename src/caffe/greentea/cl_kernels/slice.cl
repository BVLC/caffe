#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(slice,Dtype)(const int_tp nthreads,
                                    __global const Dtype* in_data,
                                    const int forward, const int_tp num_slices,
                                    const int_tp slice_size,
                                    const int_tp bottom_slice_axis,
                                    const int_tp top_slice_axis,
                                    const int_tp offset_slice_axis,
                                    __global Dtype* out_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp total_slice_size = slice_size * top_slice_axis;
    const int_tp slice_num = index / total_slice_size;
    const int_tp slice_index = index % total_slice_size;
    const int_tp bottom_index = slice_index
        + (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
    if (forward == 1) {
      out_data[index] = in_data[bottom_index];
    } else {
      out_data[bottom_index] = in_data[index];
    }
  }
}
