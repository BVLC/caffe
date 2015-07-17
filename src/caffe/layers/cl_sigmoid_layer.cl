__kernel void SigmoidForward( const int n, __global const Dtype* in, __global Dtype* out) 
{ 
  OCL_KERNEL_LOOP(index, n) 
  {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}
    
__kernel void SigmoidBackward(const int n, __global const Dtype* in_diff,
    __global const Dtype* out_data, __global Dtype* out_diff) {
  OCL_KERNEL_LOOP(index, n) 
  {
    const Dtype sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}
               