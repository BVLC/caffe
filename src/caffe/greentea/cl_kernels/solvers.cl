#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(ada_delta_update,Dtype)(int_tp N, __global Dtype* g,
                                               __global Dtype* h,
                                               __global Dtype* h2,
                                               Dtype momentum,
                                               Dtype delta,
                                               Dtype local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    Dtype gi = g[i];
    Dtype hi = h[i] = momentum * h[i] + (1.0 - momentum) * gi * gi;
    gi = gi * sqrt((h2[i] + delta) / (hi + delta));
    h2[i] = momentum * h2[i] + (1.0 - momentum) * gi * gi;
    g[i] = local_rate * gi;
  }
}

__kernel void TEMPLATE(ada_grad_update,Dtype)(int_tp N, __global Dtype* g,
                                              __global Dtype* h,
                                              Dtype delta,
                                              Dtype local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    Dtype gi = g[i];
    Dtype hi = h[i] = h[i] + gi * gi;
    g[i] = local_rate * gi / (sqrt(hi) + delta);
  }
}

__kernel void TEMPLATE(adam_update,Dtype)(int_tp N, __global Dtype* g,
                                          __global Dtype* m,
                                          __global Dtype* v,
                                          Dtype beta1,
                                          Dtype beta2,
                                          Dtype eps_hat,
                                          Dtype corrected_local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    Dtype gi = g[i];
    Dtype mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
    Dtype vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
    g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}


__kernel void TEMPLATE(nesterov_update,Dtype)(int_tp N, __global Dtype* g,
                                              __global Dtype* h,
                                              Dtype momentum,
                                              Dtype local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    Dtype hi = h[i];
    Dtype hi_new = h[i] = momentum * hi + local_rate * g[i];
    g[i] = (1 + momentum) * hi_new - momentum * hi;
  }
}

__kernel void TEMPLATE(rms_prop_update,Dtype)(int_tp N, __global Dtype* g,
                                              __global Dtype* h,
                                              Dtype rms_decay,
                                              Dtype delta,
                                              Dtype local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    Dtype gi = g[i];
    Dtype hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;
    g[i] = local_rate * g[i] / (sqrt(hi) + delta);
  }
}

__kernel void TEMPLATE(sgd_update,Dtype)(int_tp N, __global Dtype* g,
                                         __global Dtype* h,
                                         Dtype momentum,
                                         Dtype local_rate) {
  for (int_tp i = get_global_id(0); i < N; i += get_global_size(0)) {
    g[i] = h[i] = momentum * h[i] + local_rate * g[i];
  }
}
