// Copyright 2014 BVLC and contributors.
// TanH neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype exp2x;
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    exp2x = exp(2*bottom_data[i]);
    top_data[i] = (exp2x - Dtype(1))/(exp2x + Dtype(1));
  }
  return Dtype(0);
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    Dtype exp2x;
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      exp2x = exp(2*bottom_data[i]);
      tanhx = (exp2x - Dtype(1))/(exp2x + Dtype(1));
      bottom_diff[i] = top_diff[i] * (1 - tanhx*tanhx);
    }
  }
}

INSTANTIATE_CLASS(TanHLayer);

}  // namespace caffe
