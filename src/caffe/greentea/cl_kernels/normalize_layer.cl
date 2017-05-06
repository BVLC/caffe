#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif
 
__kernel void TEMPLATE(DivBsx, Dtype)(const int nthreads,
    __global const Dtype* A, const int A_off, __global const Dtype* v, const int v_off, const int rows, const int cols,
    __global Dtype* B, const int B_off) {

  for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {
    int c = index % cols;
    B[index+B_off] = A[index+A_off] / v[c+v_off];
  }
}

__kernel void TEMPLATE(MulBsx, Dtype)(const int nthreads, __global Dtype* A, const int A_off,
    __global Dtype* v, const int rows, const int cols, int trans,
    __global Dtype* B, const int B_off) {
  for (int index = get_global_id(0); index < nthreads; index += get_global_size(0)) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == 0) {
      B[index+B_off] = A[index+A_off] * v[c];
    } else {
      B[index+B_off] = A[index+A_off] * v[r];
    }
  }
}



