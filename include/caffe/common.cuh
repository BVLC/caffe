// Copyright 2014 George Papandreou

#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>

// CUDA: atomicAdd is not defined for doubles
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif
