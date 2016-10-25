#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}

// Kernel for varying momentum
template <typename Dtype>
__global__ void SGDUpdate_VM(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    g[i] = h[i] = (Dtype(-1.0)*momentum*h[i]) + ((Dtype(1.) - momentum)*local_rate*g[i]);
  }
}

template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate, bool varying_momentum) {
  if(varying_momentum) {
    SGDUpdate_VM<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
    CUDA_POST_KERNEL_CHECK;
  }
  else {
    SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, g, h, momentum, local_rate);
    CUDA_POST_KERNEL_CHECK;
  }  
}

template void sgd_update_gpu<float>(int, float*, float*, float, float, bool);
template void sgd_update_gpu<double>(int, double*, double*, double, double, bool);

}  // namespace caffe
