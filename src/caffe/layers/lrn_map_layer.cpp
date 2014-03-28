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
  // Set up conv layer to have N filters with N groups, all of which are 1's.
  // (With #filters == #groups, each filter looks at exactly 1 input channel,
  // which is what we want for this layer type.)
  // Output of conv layer gives us the neighborhood response.
  conv_top_vec_.clear();
  conv_top_vec_.push_back(&conv_output_);
  LayerParameter conv_param;
  conv_param.mutable_convolution_param()->set_pad(pre_pad);
  conv_param.mutable_convolution_param()->set_kernel_size(size_);
  conv_param.mutable_convolution_param()->set_num_output(channels);
  conv_param.mutable_convolution_param()->set_group(channels);
  conv_param.mutable_convolution_param()->set_bias_term(false);
  conv_layer_.reset(new ConvolutionLayer<Dtype>(conv_param));
  conv_layer_->SetUp(square_top_vec_, &conv_top_vec_);
  CHECK_EQ(conv_output_.num(), num);
  CHECK_EQ(conv_output_.channels(), channels);
  CHECK_EQ(conv_output_.height(), height);
  CHECK_EQ(conv_output_.width(), width);
  FillerParameter one_filler_param;
  one_filler_param.set_value(1);
  ConstantFiller<Dtype> one_filler(one_filler_param);
  one_filler.Fill(conv_layer_->blobs()[0].get());
  // Set up power layer to compute (1 + alpha/N^2 s)^-beta, where s is the sum
  // of a squared neighborhood (as output by the conv layer).
  power_top_vec_.clear();
  power_top_vec_.push_back(&power_output_);
  LayerParameter power_param;
  power_param.mutable_power_param()->set_power(-beta);
  power_param.mutable_power_param()->set_scale(alpha / (size_ * size_));
  power_param.mutable_power_param()->set_shift(Dtype(1));
  power_layer_.reset(new PowerLayer<Dtype>(power_param));
  power_layer_->SetUp(conv_top_vec_, &power_top_vec_);
  CHECK_EQ(power_output_.num(), num);
  CHECK_EQ(power_output_.channels(), channels);
  CHECK_EQ(power_output_.height(), height);
  CHECK_EQ(power_output_.width(), width);
  // Set up a product layer to compute outputs by multiplying inputs by scale.
  product_bottom_vec_.clear();
  product_bottom_vec_.push_back(bottom[0]);
  product_bottom_vec_.push_back(&power_output_);
  product_top_vec_.clear();
  product_top_vec_.push_back((*top)[0]);
  LayerParameter product_param;
  product_layer_.reset(new EltwiseProductLayer<Dtype>(product_param));
  product_layer_->SetUp(product_bottom_vec_, &product_top_vec_);
  CHECK_EQ((*top)[0]->num(), num);
  CHECK_EQ((*top)[0]->channels(), channels);
  CHECK_EQ((*top)[0]->height(), height);
  CHECK_EQ((*top)[0]->width(), width);
}

template <typename Dtype>
Dtype LRNMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* square_bottom_data = square_input_.mutable_cpu_data();
  caffe_copy(count, bottom_data, square_bottom_data);
  square_layer_->Forward(square_bottom_vec_, &square_top_vec_);
  conv_layer_->Forward(square_top_vec_, &conv_top_vec_);
  power_layer_->Forward(conv_top_vec_, &power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, &product_top_vec_);
  return Dtype(0.);
}

template <typename Dtype>
void LRNMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    product_layer_->Backward(product_top_vec_, true, &product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, true, &conv_top_vec_);
    conv_layer_->Backward(conv_top_vec_, true, &square_top_vec_);
    square_layer_->Backward(square_top_vec_, true, &square_bottom_vec_);
    const int count = (*bottom)[0]->count();
    const Dtype* scale_diff = square_input_.cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    caffe_axpy(count, Dtype(1), scale_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(LRNMapLayer);

}  // namespace caffe
