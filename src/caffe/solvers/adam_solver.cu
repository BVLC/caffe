#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdamUpdate(int N, int t, Dtype* g, Dtype* m, Dtype* v,
                           Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate,
                           bool amsgrad, bool rectified) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi_old = v[i];
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    if (amsgrad) {
      if (vi < vi_old)
        v[i] = vi = vi_old;
    }
    if (!rectified)
      g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
    else
      {
        Dtype rho_inf = 2.0/(1.0-beta2) - 1.0;
        Dtype rho_t = rho_inf - 2.0 * t * pow(beta2,t)/(1.0-pow(beta2,t)) ;
        if (rho_t > 4.0)
          {
            Dtype r_t = sqrt( (rho_t-4.0) * (rho_t-2.0) * rho_inf
                              / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t);

            g[i] = corrected_local_rate * mi * r_t / (sqrt(vi) + eps_hat);
          }
        else
          {
            g[i] = corrected_local_rate * mi;
          }
      }

  }
}


  template <typename Dtype>
  __global__ void AdamUpdateDecoupledWD(int N, int t, Dtype* g, Dtype* m, Dtype* v, const Dtype* param,
                                        Dtype beta1, Dtype beta2, Dtype eps_hat,
                                        Dtype corrected_local_rate,  Dtype nu_lambda, bool amsgrad,
                                        bool rectified) {
    CUDA_KERNEL_LOOP(i, N) {
      float gi = g[i];
      float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
      float vi_old = v[i];
      float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
      if (amsgrad) {
        if (vi < vi_old)
          v[i] = vi = vi_old;
      }
      if (!rectified)
        g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat) + param[i] * nu_lambda;
      else
        {
          Dtype rho_inf = 2.0/(1.0-beta2) - 1.0;
          Dtype rho_t = rho_inf - 2.0 * t * pow(beta2,t)/(1.0-pow(beta2,t)) ;
          if (rho_t > 4.0)
            {
              Dtype r_t = sqrt( (rho_t-4.0) * (rho_t-2.0) * rho_inf
                                / (rho_inf - 4.0) / (rho_inf - 2.0) / rho_t);
              g[i] = corrected_local_rate * mi * r_t / (sqrt(vi) + eps_hat) + param[i] * nu_lambda;
            }
          else
            {
              g[i] = corrected_local_rate * mi + param[i] * nu_lambda;
            }
        }
    }
  }

  template <typename Dtype>
  void adam_update_gpu(int N, int t, Dtype* g, Dtype* m, Dtype* v, const Dtype* param, Dtype beta1,
                     Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate, Dtype nu_lambda,
                     bool amsgrad, bool decoupled_wd, bool rectified) {
  if (!decoupled_wd)
    {
    AdamUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, t, g, m, v, beta1, beta2, eps_hat, corrected_local_rate, amsgrad, rectified);
  CUDA_POST_KERNEL_CHECK;
    }
  else{
    AdamUpdateDecoupledWD<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
     N, t, g, m, v, param, beta1, beta2, eps_hat, corrected_local_rate, nu_lambda, amsgrad, rectified);
  CUDA_POST_KERNEL_CHECK;
  }
}
  template void adam_update_gpu<float>(int, int, float*, float*, float*, const float*,
                                       float, float, float, float, float, bool, bool, bool);
  template void adam_update_gpu<double>(int, int, double*, double*, double*, const double*,
                                        double, double, double, double, double, bool, bool, bool);

}  // namespace caffe
