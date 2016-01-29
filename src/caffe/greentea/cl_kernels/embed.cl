#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(embed_forward,Dtype)(const int_tp nthreads,
                                            __global const Dtype* bottom_data,
                                            __global const Dtype* weight,
                                            const int_tp M, const int_tp N,
                                            const int_tp K,
                                            __global Dtype* top_data) {
  for (int_tp top_index = get_global_id(0); top_index < nthreads;
      top_index += get_global_size(0)) {
      const int_tp n = top_index / N;
      const int_tp d = top_index % N;
      const int_tp index = (int_tp)(bottom_data[n]);
      const int_tp weight_index = index * N + d;
      top_data[top_index] = weight[weight_index];
    }
  }

// atomic_add from: http://suhorukov.blogspot.com/2011/12/opencl-11-atomic-operations-on-floating.html
#if (TYPE == TYPE_FLOAT)
inline void TEMPLATE(atomic_add,Dtype)(volatile __global Dtype *source, const Dtype operand) {
    union {
        uint_tp intVal;
        Dtype floatVal;
    } newVal;
    union {
        uint_tp intVal;
        Dtype floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void TEMPLATE(embed_backward,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,
    __global const Dtype* top_diff, const int_tp M, const int_tp N, const int_tp K,
    __global Dtype* weight_diff) {
  for (int_tp top_index = get_global_id(0); top_index < nthreads;
      top_index += get_global_size(0)) {
    const int_tp n = top_index / N;
    const int_tp d = top_index % N;
    const int_tp index = (int_tp)(bottom_data[n]);
    const int_tp weight_index = index * N + d;

    TEMPLATE(atomic_add,Dtype)((weight_diff + weight_index), *(top_diff + top_index));
  }
}
#endif

#if (TYPE == TYPE_DOUBLE)
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

__kernel void TEMPLATE(embed_backward,Dtype)(const int_tp nthreads, __global const Dtype* bottom_data,
    __global const Dtype* top_diff, const int_tp M, const int_tp N, const int_tp K,
    __global Dtype* weight_diff) {
  for (int_tp top_index = get_global_id(0); top_index < nthreads;
      top_index += get_global_size(0)) {
    const int_tp n = top_index / N;
    const int_tp d = top_index % N;
    const int_tp index = (int_tp)(bottom_data[n]);
    const int_tp weight_index = index * N + d;

    TEMPLATE(atomic_add,Dtype)((weight_diff + weight_index), *(top_diff + top_index));
  }
}
#endif
#endif
