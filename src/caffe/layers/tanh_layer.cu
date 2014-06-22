// Copyright 2014 BVLC and contributors.
// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TanHForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype exp2x = exp(2*in[index]);
    out[index] = (exp2x - Dtype(1))/(exp2x + Dtype(1));
  }
}

template <typename Dtype>
Dtype TanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TanHForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  return Dtype(0);
}

template <typename Dtype>
__global__ void TanHBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype exp2x = exp(2*in_data[index]);
    Dtype tanhx = (exp2x - Dtype(1))/(exp2x + Dtype(1));
    out_diff[index] = in_diff[index] * (1 - tanhx*tanhx);
  }
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    TanHBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(TanHLayer);


}  // namespace caffe
