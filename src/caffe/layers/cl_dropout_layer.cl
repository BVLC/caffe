__kernel void DropoutForward(const int count, __global Dtype* in,
    __global unsigned int* mask, const unsigned int threshold, 
    const Dtype scale, __global Dtype* out) {
  OCL_KERNEL_LOOP(index, count) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

__kernel void DropoutBackward(const int count, __global Dtype* in_diff,
    __global unsigned int* mask, const unsigned int threshold, 
    const Dtype scale, __global Dtype* out_diff) {
  OCL_KERNEL_LOOP(index, count) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}