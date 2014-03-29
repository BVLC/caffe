// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LRNMapLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) <<
      "Local Response Normalization Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) <<
      "Local Response Normalization Layer takes a single blob as output.";
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int size_ = this->layer_param_.lrn_map_param().local_size();
  const Dtype pre_pad = (size_ - 1) / 2;
  const Dtype alpha = this->layer_param_.lrn_map_param().alpha();
  const Dtype beta = this->layer_param_.lrn_map_param().beta();
  // Set up split layer to use inputs in the numerator and denominator.
  split_top_vec_.clear();
  split_top_vec_.push_back(bottom[0]);
  split_top_vec_.push_back(&square_input_);
  LayerParameter split_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_param));
  split_layer_->SetUp(bottom, &split_top_vec_);
  // Set up square layer to square the inputs.
  square_input_.Reshape(num, channels, height, width);
  square_bottom_vec_.clear();
  square_top_vec_.clear();
  square_bottom_vec_.push_back(&square_input_);
  square_top_vec_.push_back(&square_output_);
  LayerParameter square_param;
  square_param.mutable_power_param()->set_power(Dtype(2));
  square_layer_.reset(new PowerLayer<Dtype>(square_param));
  square_layer_->SetUp(square_bottom_vec_, &square_top_vec_);
  CHECK_EQ(square_output_.num(), num);
  CHECK_EQ(square_output_.channels(), channels);
  CHECK_EQ(square_output_.height(), height);
  CHECK_EQ(square_output_.width(), width);
  // Output of pool layer gives us the neighborhood response.
  pool_top_vec_.clear();
  pool_top_vec_.push_back(&pool_output_);
  LayerParameter pool_param;
  pool_param.mutable_pooling_param()->set_pool(PoolingParameter_PoolMethod_AVE);
  pool_param.mutable_pooling_param()->set_pad(pre_pad);
  pool_param.mutable_pooling_param()->set_kernel_size(size_);
  pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
  pool_layer_->SetUp(square_top_vec_, &pool_top_vec_);
  CHECK_EQ(pool_output_.num(), num);
  CHECK_EQ(pool_output_.channels(), channels);
  CHECK_EQ(pool_output_.height(), height);
  CHECK_EQ(pool_output_.width(), width);
  // Set up power layer to compute (1 + alpha/N^2 s)^-beta, where s is the sum
  // of a squared neighborhood (as output by the conv layer).
  power_top_vec_.clear();
  power_top_vec_.push_back(&power_output_);
  LayerParameter power_param;
  power_param.mutable_power_param()->set_power(-beta);
  power_param.mutable_power_param()->set_scale(alpha);
  power_param.mutable_power_param()->set_shift(Dtype(1));
  power_layer_.reset(new PowerLayer<Dtype>(power_param));
  power_layer_->SetUp(pool_top_vec_, &power_top_vec_);
  CHECK_EQ(power_output_.num(), num);
  CHECK_EQ(power_output_.channels(), channels);
  CHECK_EQ(power_output_.height(), height);
  CHECK_EQ(power_output_.width(), width);
  // Set up a product layer to compute outputs by multiplying inputs by the
  // demoninator computed by the power layer.
  product_bottom_vec_.clear();
  product_bottom_vec_.push_back(bottom[0]);
  product_bottom_vec_.push_back(&power_output_);
  LayerParameter product_param;
  product_layer_.reset(new EltwiseProductLayer<Dtype>(product_param));
  product_layer_->SetUp(product_bottom_vec_, top);
  CHECK_EQ((*top)[0]->num(), num);
  CHECK_EQ((*top)[0]->channels(), channels);
  CHECK_EQ((*top)[0]->height(), height);
  CHECK_EQ((*top)[0]->width(), width);
}

template <typename Dtype>
Dtype LRNMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  split_layer_->Forward(bottom, &split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, &square_top_vec_);
  pool_layer_->Forward(square_top_vec_, &pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, &power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
  return Dtype(0.);
}

template <typename Dtype>
void LRNMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
