// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype LRNMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* square_bottom_data = square_input_.mutable_gpu_data();
  caffe_gpu_copy(count, bottom_data, square_bottom_data);
  square_layer_->Forward(square_bottom_vec_, &square_top_vec_);
  pool_layer_->Forward(square_top_vec_, &pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, &power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, &product_top_vec_);
  return Dtype(0.);
}

template <typename Dtype>
void LRNMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    product_layer_->Backward(product_top_vec_, true, &product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, true, &pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, true, &square_top_vec_);
    square_layer_->Backward(square_top_vec_, true, &square_bottom_vec_);
    const int count = (*bottom)[0]->count();
    const Dtype* scale_diff = square_input_.gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    caffe_gpu_axpy(count, Dtype(1), scale_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(LRNMapLayer);

}  // namespace caffe
