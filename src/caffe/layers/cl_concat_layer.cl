__kernel void ConcatForward(const int nthreads, __global Dtype* in_data,
    const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, __global Dtype* out_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[index];
  }
}

__kernel void ConcatBackward(const int nthreads, __global Dtype* in_data,
    const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, __global Dtype* out_data) {
  OCL_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[index] = in_data[top_index];
  }
}
