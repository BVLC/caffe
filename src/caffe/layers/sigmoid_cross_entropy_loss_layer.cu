// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  return loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // First, compute the diff
  const int count = (*bottom)[0]->count();
  const int num = (*bottom)[0]->num();
  const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
  const Dtype* target = (*bottom)[1]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  caffe_gpu_copy(count, sigmoid_output_data, bottom_diff);
  caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
  // Scale down gradient
  caffe_gpu_scal(count, Dtype(1) / num, bottom_diff);
}

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
