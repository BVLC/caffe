#include <cmath>
#include <vector>

#include "caffe/layers/swish_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SwishForward(const int n, const Dtype* in, Dtype* sigmoid_x,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    sigmoid_x[index] = 0.5 * tanh(0.5 * in[index]) + 0.5;
    out[index] = in[index] * sigmoid_x[index];
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* sigmoid_x_data = this->sigmoid_x_.mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SwishForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, sigmoid_x_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SwishBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, const Dtype* sigmoid_x_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype swish_x = out_data[index];
    out_diff[index] = in_diff[index] * (swish_x + sigmoid_x_data[index]
        * (1 - swish_x));
  }
}

template <typename Dtype>
void SwishLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* sigmoid_x_data = this->sigmoid_x_.gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SwishBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_data, sigmoid_x_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwishLayer);

}  // namespace caffe
