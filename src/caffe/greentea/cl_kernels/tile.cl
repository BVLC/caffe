#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif


__kernel void TEMPLATE(tile,Dtype)(const int nthreads, __global const Dtype* bottom_data,
                                   const int tile_size, const int num_tiles,
                                   const int bottom_tile_axis,
                                   __global Dtype* top_data) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int d = index % tile_size;
    const int b = (index / tile_size / num_tiles) % bottom_tile_axis;
    const int n = index / tile_size / num_tiles / bottom_tile_axis;
    const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}


__kernel void TEMPLATE(tile_backward,Dtype)(const int nthreads,
                                            __global const Dtype* top_diff,
                                            const int tile_size,
                                            const int num_tiles,
                                            const int bottom_tile_axis,
                                            __global Dtype* bottom_diff) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int d = index % tile_size;
    const int b = (index / tile_size) % bottom_tile_axis;
    const int n = index / tile_size / bottom_tile_axis;
    bottom_diff[index] = 0;
    int top_index = (n * num_tiles * bottom_tile_axis + b) * tile_size + d;
    for (int t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += bottom_tile_axis * tile_size;
    }
  }
}
