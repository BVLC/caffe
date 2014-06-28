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
void SigmoidCrossEntropyLossLayer<Dtype>::FurtherSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->const_data();
  const Dtype* target = bottom[1]->const_data();
  Dtype loss = 0;
  for (int i = 0; i < count; ++i) {
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
  }
  if (top->size() == 1) {
    (*top)[0]->mutable_data()[0] = loss / num;
  }
  return loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = (*bottom)[0]->count();
    const int num = (*bottom)[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->const_data();
    const Dtype* target = (*bottom)[1]->const_data();
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    this->device_->sub(count, sigmoid_output_data, target, bottom_diff);
    // Scale down gradient
    this->device_->scal(count, Dtype(1) / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
