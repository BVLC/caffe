__kernel void ThresholdForward(const int n, const Dtype threshold,
    __global const Dtype* in, __global Dtype* out) {
  OCL_KERNEL_LOOP(index, n) {
    out[index] = in[index] > threshold ? 1 : 0;
  }
}
