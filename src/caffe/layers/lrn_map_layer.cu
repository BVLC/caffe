// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype LRNMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  split_layer_->Forward(bottom, &split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, &square_top_vec_);
  pool_layer_->Forward(square_top_vec_, &pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, &power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
  return Dtype(0.);
}

template <typename Dtype>
void LRNMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    product_layer_->Backward(top, true, &product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, true, &pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, true, &square_top_vec_);
    square_layer_->Backward(square_top_vec_, true, &square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, true, bottom);
  }
}

INSTANTIATE_CLASS(LRNMapLayer);

}  // namespace caffe
