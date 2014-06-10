// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      caffe_gpu_axpby(
          (*bottom)[i]->count(),              // count
          sign / (*bottom)[i]->num(),         // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
