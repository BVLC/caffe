#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(mul,Dtype)(const int n, __global const Dtype* a,
                                  const int offa,
                                  __global Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] * b[index + offb];
  }
}

__kernel void TEMPLATE(div,Dtype)(const int n, __global const Dtype* a,
                                  const int offa,
                                  __global Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = a[index + offa] / b[index + offb];
  }
}

__kernel void TEMPLATE(add_scalar,Dtype)(const int N, const Dtype alpha,
__global Dtype* Y,
                                         const int offY) {
  for (int index = get_global_id(0); index < N; index += get_global_size(0)) {
    Y[offY + index] += alpha;
  }
}

__kernel void TEMPLATE(add,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global const Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] + b[offb + index];
  }
}

__kernel void TEMPLATE(sub,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global const Dtype* b,
                                  const int offb, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = a[offa + index] - b[offb + index];
  }
}

__kernel void TEMPLATE(abs,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = fabs((Dtype)(a[offa + index]));
  }
}

__kernel void TEMPLATE(exp,Dtype)(const int n, __global const Dtype* a,
                                  const int offa, __global Dtype* y,
                                  const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = exp(a[offa + index]);
  }
}

__kernel void TEMPLATE(powx,Dtype)(const int n, __global const Dtype* a,
                                   const int offa, Dtype alpha,
                                   __global Dtype* y,
                                   const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[offy + index] = pow(a[offa + index], alpha);
  }
}

__kernel void TEMPLATE(sign,Dtype)(const int n, __global const Dtype* x,
                                   const int offx, __global Dtype* y,
                                   const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = (0.0 < x[index + offx])
        - (x[index + offx] < 0.0);
  }
}

__kernel void TEMPLATE(sgnbit,Dtype)(const int n, __global const Dtype* x,
                                     const int offx, __global Dtype* y,
                                     const int offy) {
  for (int index = get_global_id(0); index < n; index += get_global_size(0)) {
    y[index + offy] = signbit(x[index + offx]);
  }
}
