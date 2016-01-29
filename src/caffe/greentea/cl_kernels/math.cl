#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(mul,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa,
                                  __global Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] * b[index + offb];
  }
}

__kernel void TEMPLATE(div,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa,
                                  __global Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] / b[index + offb];
  }
}

__kernel void TEMPLATE(add_scalar,Dtype)(const int_tp N, const Dtype alpha,
__global Dtype* Y,
                                         const int_tp offY) {
  for (int_tp index = get_global_id(0); index < N; index += get_global_size(0)) {
    Y[offY + index] += alpha;
  }
}

__kernel void TEMPLATE(add,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global const Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] + b[offb + index];
  }
}

__kernel void TEMPLATE(sub,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global const Dtype* b,
                                  const int_tp offb, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] - b[offb + index];
  }
}

__kernel void TEMPLATE(abs,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = fabs((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(exp,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = exp(a[offa + index]);
  }
}

__kernel void TEMPLATE(log,Dtype)(const int_tp n, __global const Dtype* a,
                                  const int_tp offa, __global Dtype* y,
                                  const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = log((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(powx,Dtype)(const int_tp n, __global const Dtype* a,
                                   const int_tp offa, Dtype alpha,
                                   __global Dtype* y,
                                   const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    if(alpha == 2.0) {
      y[offy + index] = pow((Dtype)fabs(a[offa + index]), (Dtype)alpha);
    } else {
      y[offy + index] = pow((Dtype)a[offa + index], (Dtype)alpha);
    }
  }
}

__kernel void TEMPLATE(sign,Dtype)(const int_tp n, __global const Dtype* x,
                                   const int_tp offx, __global Dtype* y,
                                   const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = (0.0 < x[index + offx])
        - (x[index + offx] < 0.0);
  }
}

__kernel void TEMPLATE(sgnbit,Dtype)(const int_tp n, __global const Dtype* x,
                                     const int_tp offx, __global Dtype* y,
                                     const int_tp offy) {
  for (int_tp index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = signbit(x[index + offx]);
  }
}
