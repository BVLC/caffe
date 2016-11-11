#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(smooth_l1_loss_forward,Dtype)(const int_tp n, __global const Dtype* in, __global Dtype* out,
    Dtype sigma2) {

  for (int_tp index = get_global_id(0); index < n;
      index += get_global_size(0)) {
    Dtype val = in[index];
    //Dtype abs_val = abs(val);
    if ((val < 1.0 / sigma2) && (-val > -(1.0 / sigma2))) {
      out[index] = 0.5 * val * val * sigma2;
    } else {
      if (val<0.0){
        out[index] = -val - 0.5 / sigma2;
      } else {
        out[index] = val - 0.5 / sigma2;
      }
    }
  }
}

__kernel void TEMPLATE(softmax_loss_backward,Dtype)(const int_tp n, __global const Dtype* in, __global Dtype* out,
    Dtype sigma2) {
  for (int_tp index = get_global_id(0); index < n; index +=
      get_global_size(0)) {
    Dtype val = in[index];
    //Dtype abs_val = abs(val);
    if ((val < 1.0 / sigma2) && (-val > -(1.0 / sigma2))) {
      out[index] = sigma2 * val;
    } else {
      out[index] = ((Dtype)(0) < val) - (val < (Dtype)(0));
    }
  }
}

