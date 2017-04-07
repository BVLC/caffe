#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    float gi = g[i];
    float hi = h[i] = h[i] + gi*gi;
    g[i] = local_rate * gi / (sqrt(hi) + delta);
  }
}
template <typename Dtype>
void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta,
    Dtype local_rate) {
  AdaGradUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, delta, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void adagrad_update_gpu<float>(int, float*, float*, float, float);
template void adagrad_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
