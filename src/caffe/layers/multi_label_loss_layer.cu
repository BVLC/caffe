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
Dtype MultiLabelLossLayer<Dtype>::Forward_gpu(
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
    if (target[i] != 0) {
    // Update the loss only if target[i] is not 0
      loss -= input_data[i] * ((target[i] > 0) - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }
  if (top->size() >= 1) {
    (*top)[0]->mutable_cpu_data()[0] = loss / num;
  }
  return loss / num;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // TODO: implement the GPU version of sigmoid_cross_entropy with -1,0,1 labels
  Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(MultiLabelLossLayer);


}  // namespace caffe
