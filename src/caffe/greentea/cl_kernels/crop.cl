#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

inline int_tp TEMPLATE(compute_uncropped_index,Dtype)(
    int_tp index,
    const int_tp ndims,
    __global const int_tp*  src_strides,
    __global const int_tp*  dst_strides,
    __global const int_tp*  offsets) {
  int_tp dest_index = index;
  int_tp src_index = 0;
  for (int_tp i = 0; i < ndims; ++i) {
      int_tp coord = dest_index / dst_strides[i];
      dest_index -= coord * dst_strides[i];
      src_index += src_strides[i] * (coord + offsets[i]);
  }
  return src_index;
}

__kernel void TEMPLATE(crop_forward,Dtype)(const int_tp nthreads,
    const int_tp ndims,
    __global const int_tp*  src_strides,
    __global const int_tp*  dst_strides,
    __global const int_tp*  offsets,
    __global const Dtype* src,
    const int_tp src_off,
    __global Dtype*  dst,
    const int_tp dst_off) {
  for (int_tp index = get_global_id(0); index < nthreads;
        index += get_global_size(0)) {
    int_tp src_index = TEMPLATE(compute_uncropped_index,Dtype)(
        index, ndims, src_strides, dst_strides, offsets);
    dst[dst_off + index] = src[src_off + src_index];
  }
}

__kernel void TEMPLATE(crop_backward,Dtype)(const int_tp nthreads,
    const int_tp ndims,
    __global const int_tp*  src_strides,
    __global const int_tp*  dst_strides,
    __global const int_tp*  offsets,
    __global Dtype*  src,
    const int_tp src_off,
    __global const Dtype* dst,
    const int_tp dst_off) {
  for (int_tp index = get_global_id(0); index < nthreads;
        index += get_global_size(0)) {
    int_tp src_index = TEMPLATE(compute_uncropped_index,Dtype)(
        index, ndims, src_strides, dst_strides, offsets);
    src[src_off + src_index] = dst[dst_off + index];
  }
}
