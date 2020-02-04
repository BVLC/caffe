#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h, const Dtype* param,
                          Dtype momentum, Dtype local_rate, Dtype lambda, Dtype nu, bool decoupled,
                          Dtype lr_dropout, const Dtype*r) {
  CUDA_KERNEL_LOOP(i, N) {
    if (lr_dropout == 1.0)
      {
        if (decoupled)
          g[i] = h[i] = momentum*h[i] + local_rate*g[i] * nu + param[i] * lambda * nu;
        else
          g[i] = h[i] = momentum*h[i] + local_rate*g[i];
        return;
      }

    if (r[i] > lr_dropout)
      {
        g[i] = h[i] = Dtype(0.0);
        return;
      }
    if (decoupled)
      g[i] = h[i] = momentum*h[i] + local_rate*g[i] * nu + param[i] * lambda * nu;
    else
      g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, const Dtype * param, Dtype momentum,
                    Dtype local_rate, Dtype lambda, Dtype nu, bool decoupled, Dtype lr_dropout,
                    const Dtype *r) {

  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
                           N, g, h, param, momentum, local_rate , nu, lambda, decoupled, lr_dropout,r);
  CUDA_POST_KERNEL_CHECK;
}
  template void sgd_update_gpu<float>(int, float*, float*, const float*, float, float, float, float, bool, float, const float*);
  template void sgd_update_gpu<double>(int, double*, double*, const double*, double, double, double, double, bool, double, const double*);

}  // namespace caffe
