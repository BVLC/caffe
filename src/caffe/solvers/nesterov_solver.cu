#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void NesterovUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float hi = h[i];
    float hi_new = h[i] = momentum * hi + local_rate * g[i];
    g[i] = (1+momentum) * hi_new - momentum * hi;
  }
}
template <typename Dtype>
void nesterov_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  NesterovUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void nesterov_update_gpu<float>(int, float*, float*, float, float);
template void nesterov_update_gpu<double>(int, double*, double*, double,
    double);

}  // namespace caffe
