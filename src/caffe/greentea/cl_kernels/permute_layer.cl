#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

#define OCL_KERNEL_LOOP(i, n)  for (int i = get_global_id(0); i < (n); i += get_global_size(0))
 
__kernel void TEMPLATE(PermuteKernel, Dtype)(const int nthreads,
    __global Dtype* bottom_data, const int forward, global int* permute_order,
    global int* old_steps, global int* new_steps, const int num_axes,
    __global Dtype* top_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
  int temp_idx = index;
  int old_idx = 0;
  for (int i = 0; i < num_axes; ++i) {
    int order = permute_order[i];
    old_idx += (temp_idx / new_steps[i]) * old_steps[order];
    temp_idx %= new_steps[i];
  }
  if (forward != 0) {
    top_data[index] = bottom_data[old_idx];
  } else {
    bottom_data[old_idx] = top_data[index];
  }
}
}



