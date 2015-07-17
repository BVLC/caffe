#define kBNLL_THRESHOLD 50.0f

__kernel void BNLLForward(const int n, __global Dtype* in, __global Dtype* out) {
  OCL_KERNEL_LOOP(index, n) {  
    if(in[index] > 0)
    	out[index] =  in[index] + log(1.0f + exp(-in[index]));
    else
        out[index] = log(1.0f + exp(in[index]));
  }
}

__kernel void BNLLBackward(const int n, __global Dtype* in_diff,
    __global Dtype* in_data, __global Dtype* out_diff) {
  OCL_KERNEL_LOOP(index, n) {
    Dtype expval = exp(min(in_data[index], (kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.f);
  }
}