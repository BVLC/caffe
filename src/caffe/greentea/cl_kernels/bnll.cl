#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(bnll_forward,Dtype)(const int n,
                                           __global const Dtype* in,
                                           __global Dtype* out) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    out[index] =
        in[index] > 0 ?
            in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));
  }
}

__kernel void TEMPLATE(bnll_backward,Dtype)(const int n,
                                            __global const Dtype* in_diff,
                                            __global const Dtype* in_data,
                                            __global Dtype* out_diff) {
  float kBNLL_THRESHOLD = 50.;
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}
