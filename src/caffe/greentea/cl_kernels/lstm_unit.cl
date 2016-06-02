#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

inline Dtype TEMPLATE(lstm_sigmoid,Dtype)(const Dtype x) {
  return (Dtype)1 / ((Dtype)1 + exp(-x));
}

inline Dtype TEMPLATE(lstm_tanh,Dtype)(const Dtype x) {
  return (Dtype)2 * TEMPLATE(lstm_sigmoid,Dtype)((Dtype)2 * x) - (Dtype)1;
}

__kernel void TEMPLATE(lstm_acts_forward,Dtype)(const int_tp nthreads, const int_tp dim,
                                __global const Dtype* X, __global Dtype* X_acts) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp x_dim = 4 * dim;
    const int_tp d = index % x_dim;
    if (d < 3 * dim) {
      X_acts[index] = TEMPLATE(lstm_sigmoid,Dtype)(X[index]);
    } else {
      X_acts[index] = TEMPLATE(lstm_tanh,Dtype)(X[index]);
    }
  }
}

__kernel void TEMPLATE(lstm_unit_forward,Dtype)(const int_tp nthreads, const int_tp dim,
    __global const Dtype* C_prev, __global const Dtype* X, __global const Dtype* cont,
    __global Dtype* C, __global Dtype* H) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp n = index / dim;
    const int_tp d = index % dim;
    __global const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = cont[n] * f * c_prev + i * g;
    C[index] = c;
    const Dtype tanh_c = TEMPLATE(lstm_tanh,Dtype)(c);
    H[index] = o * tanh_c;
  }
}

__kernel void TEMPLATE(lstm_unit_backward,Dtype)(const int_tp nthreads, const int_tp dim,
    __global const Dtype* C_prev, __global const Dtype* X, __global const Dtype* C, __global const Dtype* H,
    __global const Dtype* cont, __global const Dtype* C_diff, __global const Dtype* H_diff,
    __global Dtype* C_prev_diff, __global Dtype* X_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp n = index / dim;
    const int_tp d = index % dim;
    __global const Dtype* X_offset = X + 4 * dim * n;
    const Dtype i = X_offset[d];
    const Dtype f = X_offset[1 * dim + d];
    const Dtype o = X_offset[2 * dim + d];
    const Dtype g = X_offset[3 * dim + d];
    const Dtype c_prev = C_prev[index];
    const Dtype c = C[index];
    const Dtype tanh_c = TEMPLATE(lstm_tanh,Dtype)(c);
    __global Dtype* c_prev_diff = C_prev_diff + index;
    __global Dtype* X_diff_offset = X_diff + 4 * dim * n;
    __global Dtype* i_diff = X_diff_offset + d;
    __global Dtype* f_diff = X_diff_offset + 1 * dim + d;
    __global Dtype* o_diff = X_diff_offset + 2 * dim + d;
    __global Dtype* g_diff = X_diff_offset + 3 * dim + d;
    const Dtype c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    const Dtype cont_n = cont[n];
    *c_prev_diff = cont_n * c_term_diff * f;
    *i_diff = c_term_diff * g;
    *f_diff = cont_n * c_term_diff * c_prev;
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = c_term_diff * i;
  }
}

__kernel void TEMPLATE(lstm_acts_backward,Dtype)(const int_tp nthreads, const int_tp dim,
          __global const Dtype* X_acts, __global const Dtype* X_acts_diff, __global Dtype* X_diff) {
  for (int_tp index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int_tp x_dim = 4 * dim;
    const int_tp d = index % x_dim;
    const Dtype X_act = X_acts[index];
    if (d < 3 * dim) {
      X_diff[index] = X_acts_diff[index] * X_act * ((Dtype)1 - X_act);
    } else {
      X_diff[index] = X_acts_diff[index] * ((Dtype)1 - X_act * X_act);
    }
  }
}
