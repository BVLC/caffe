#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(embed_forward,Dtype)(const int nthreads,
                                            __global const Dtype* bottom_data,
                                            __global const Dtype* weight,
                                            const int M, const int N,
                                            const int K,
                                            __global Dtype* top_data) {
  for (int top_index = get_global_id(0); top_index < nthreads;
      top_index += get_global_size(0)) {
      const int n = top_index / N;
      const int d = top_index % N;
      const int index = (int)(bottom_data[n]);
      const int weight_index = index * N + d;
      top_data[top_index] = weight[weight_index];
    }
  }

// atomic_add from: http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
#if (Dtype == float)
inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {
    union {
        unsigned int intVal;
        Dtype floatVal;
    } newVal;
    union {
        unsigned int intVal;
        Dtype floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
#else
#ifdef ATOMICS_64_AVAILABLE
inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {
    union {
        unsigned long intVal;
        Dtype floatVal;
    } newVal;
    union {
        unsigned long intVal;
        Dtype floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atom_cmpxchg((volatile __global unsigned long *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}
#endif
#endif

__kernel void TEMPLATE(embed_backward,Dtype)(const int nthreads, __global const Dtype* bottom_data,
    __global const Dtype* top_diff, const int M, const int N, const int K,
    __global Dtype* weight_diff) {
  for (int top_index = get_global_id(0); top_index < nthreads;
      top_index += get_global_size(0)) {
    const int n = top_index / N;
    const int d = top_index % N;
    const int index = (int)(bottom_data[n]);
    const int weight_index = index * N + d;

    TEMPLATE(atomic_add,Dtype)((weight_diff + weight_index), *(top_diff + top_index));
  }
}
