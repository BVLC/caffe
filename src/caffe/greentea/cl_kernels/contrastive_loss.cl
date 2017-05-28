#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(cll_backward,Dtype)(const int_tp count, const int_tp channels,
                            const Dtype margin, const Dtype alpha, __global const Dtype* y,
                            __global const Dtype* diff, __global const Dtype* dist_sq,
                            __global Dtype *bottom_diff) {
  for (int_tp i = get_global_id(0); i < count;
      i += get_global_size(0)) {
    int_tp n = i / channels;  // the num index, to access y and dist_sq
    if (trunc(y[n]) != 0.) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist = 0.;
      Dtype beta = 0.;
      Dtype dist = sqrt(dist_sq[n]);
      mdist = (margin - dist);
      beta = -alpha * mdist / (dist + 1e-4) * diff[i];
      if (mdist > 0.) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

__kernel void TEMPLATE(cll_backward_legacy,Dtype)(const int count, const int channels,
    const Dtype margin, const Dtype alpha, __global Dtype* y,
    __global Dtype* diff, __global Dtype* dist_sq,
    __global Dtype* bottom_diff) {
    for (int_tp i = get_global_id(0); i < count;
      i += get_global_size(0)) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (trunc(y[n]) != 0.) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      Dtype mdist = 0.;
      Dtype beta = 0.;
      mdist = (margin - dist_sq[n]);
      beta = -alpha;
      if (mdist > 0.) {
        bottom_diff[i] = beta;
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}
