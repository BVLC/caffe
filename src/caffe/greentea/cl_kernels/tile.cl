#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif


__kernel void TEMPLATE(tile,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,
                                   const int_tp tile_size, const int_tp num_tiles,
                                   const int_tp bottom_tile_axis,
                                   __global Dtype* top_data) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp d = index % tile_size;
    const int_tp b = (index / tile_size / num_tiles) % bottom_tile_axis;
    const int_tp n = index / tile_size / num_tiles / bottom_tile_axis;
    const int_tp bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}


__kernel void TEMPLATE(tile_backward,Dtype)(const int_tp nthreads,
                                            __global const Dtype* top_diff,
                                            const int_tp tile_size,
                                            const int_tp num_tiles,
                                            const int_tp bottom_tile_axis,
                                            __global Dtype* bottom_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp d = index % tile_size;
    const int_tp b = (index / tile_size) % bottom_tile_axis;
    const int_tp n = index / tile_size / bottom_tile_axis;
    bottom_diff[index] = 0;
    int_tp top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;
    for (int_tp t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += bottom_tile_axis * tile_size;
    }
  }
}
