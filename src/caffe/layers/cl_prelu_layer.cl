__kernel void PReLUForward(const int n, const int channels, const int dim,
    __global Dtype* in, __global Dtype* out, __global Dtype* slope_data,
    const int div_factor) {
  OCL_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}

__kernel void PReLUBackward(const int n, const int channels, const int dim,
    __global Dtype* in_diff, __global Dtype* in_data, __global Dtype* out_diff,
    __global Dtype* slope_data, const int div_factor) {
  OCL_KERNEL_LOOP(index, n) {
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}

__kernel void PReLUParamBackward(const int n, __global Dtype* in_diff,
    const int in_diff_offset, __global Dtype* in_data, const int in_data_offset,
    __global Dtype* out_diff) {
  OCL_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[in_diff_offset + index] *
                      in_data[in_data_offset + index] *
                      (in_data[in_data_offset + index] <= 0);
  }
}
