#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v,
                           Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate,
                           bool amsgrad) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi_old = v[i];
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    if (amsgrad) {
      if (vi < vi_old)
        v[i] = vi = vi_old;
    }
    g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}


  template <typename Dtype>
  __global__ void AdamUpdateDecoupledWD(int N, Dtype* g, Dtype* m, Dtype* v, const Dtype* param,
                                        Dtype beta1, Dtype beta2, Dtype eps_hat,
                                        Dtype corrected_local_rate,  Dtype nu_lambda, bool amsgrad) {
    CUDA_KERNEL_LOOP(i, N) {
      float gi = g[i];
      float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
      float vi_old = v[i];
      float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
      if (amsgrad) {
        if (vi < vi_old)
          v[i] = vi = vi_old;
      }
      g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat) + param[i] * nu_lambda;
    }
  }

  template <typename Dtype>
  void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, const Dtype* param, Dtype beta1,
                     Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype nu_lambda,
                     bool amsgrad, bool decoupled_wd) {
  if (!decoupled_wd)
    {
    AdamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
     N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate, amsgrad);
  CUDA_POST_KERNEL_CHECK;
    }
  else{
    AdamUpdateDecoupledWD<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, param, beta1, beta2, eps_hat, corrected_local_rate, nu_lambda, amsgrad);
  CUDA_POST_KERNEL_CHECK;
  }
}
  template void adam_update_gpu<float>(int, float*, float*, float*, const float*,
                                       float, float, float, float, float, bool, bool);
  template void adam_update_gpu<double>(int, double*, double*, double*, const double*,
                                        double, double, double, double, double, bool, bool);

}  // namespace caffe
