// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype EltwiseProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
  for (int i = 2; i < bottom.size(); ++i) {
    caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    for (int i = 0; i < bottom->size(); ++i) {
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
      caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(EltwiseProductLayer);


}  // namespace caffe
