__kernel void ReLUForward(const int count, __global Dtype* in, 
    __global Dtype* out, const Dtype negative_slope) {
  OCL_KERNEL_LOOP(index, count) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

__kernel void ReLUBackward(const int count, __global Dtype* in_diff,
    __global Dtype* in_data, __global Dtype* out_diff, 
    const Dtype negative_slope) {
  OCL_KERNEL_LOOP(index, count) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0) + 
        (in_data[index] <= 0) * negative_slope);
  }
}
