#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe {

const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
__global__ void BNLLForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ?
        in[index] + log(1. + exp(-in[index])) :
        log(1. + exp(in[index]));
  }
}

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BNLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BNLLBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
    out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(BNLLLayer);


}  // namespace caffe
