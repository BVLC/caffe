#include <algorithm>

#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdaMaxUpdate(int N, Dtype* g, Dtype* m, Dtype* v,
    Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = max(v[i]*beta2 + eps_hat, abs(gi));
    g[i] = corrected_local_rate * mi / vi;
  }
}
template <typename Dtype>
void adamax_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
    Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate) {
  AdaMaxUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void adamax_update_gpu<float>(int, float*, float*, float*,
    float, float, float, float);
template void adamax_update_gpu<double>(int, double*, double*, double*,
    double, double, double, double);

}  // namespace caffe
