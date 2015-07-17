__kernel void TanHForward(const int n, __global const Dtype* in, __global Dtype* out) {
  OCL_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

__kernel void TanHBackward(const int n, __global const Dtype* in_diff,
    __global const Dtype* out_data, __global Dtype* out_diff) {
  OCL_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}
