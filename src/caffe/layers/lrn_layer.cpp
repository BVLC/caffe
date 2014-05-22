// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LRNLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) <<
      "Local Response Normalization Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) <<
      "Local Response Normalization Layer takes a single blob as output.";
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  size_ = this->layer_param_.lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param_.lrn_param().alpha();
  beta_ = this->layer_param_.lrn_param().beta();
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    (*top)[0]->Reshape(num_, channels_, height_, width_);
    scale_.Reshape(num_, channels_, height_, width_);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    {
      // Set up split_layer_ to use inputs in the numerator and denominator.
      split_top_vec_.clear();
      split_top_vec_.push_back(bottom[0]);
      split_top_vec_.push_back(&square_input_);
      LayerParameter split_param;
      split_layer_.reset(new SplitLayer<Dtype>(split_param));
      split_layer_->SetUp(bottom, &split_top_vec_);
      // Set up square_layer_ to square the inputs.
      square_input_.Reshape(num_, channels_, height_, width_);
      square_bottom_vec_.clear();
      square_top_vec_.clear();
      square_bottom_vec_.push_back(&square_input_);
      square_top_vec_.push_back(&square_output_);
      LayerParameter square_param;
      square_param.mutable_power_param()->set_power(Dtype(2));
      square_layer_.reset(new PowerLayer<Dtype>(square_param));
      square_layer_->SetUp(square_bottom_vec_, &square_top_vec_);
      CHECK_EQ(square_output_.num(), num_);
      CHECK_EQ(square_output_.channels(), channels_);
      CHECK_EQ(square_output_.height(), height_);
      CHECK_EQ(square_output_.width(), width_);
      // Set up pool_layer_ to sum over square neighborhoods of the input.
      pool_top_vec_.clear();
      pool_top_vec_.push_back(&pool_output_);
      LayerParameter pool_param;
      pool_param.mutable_pooling_param()->set_pool(
          PoolingParameter_PoolMethod_AVE);
      pool_param.mutable_pooling_param()->set_pad(pre_pad_);
      pool_param.mutable_pooling_param()->set_kernel_size(size_);
      pool_layer_.reset(new PoolingLayer<Dtype>(pool_param));
      pool_layer_->SetUp(square_top_vec_, &pool_top_vec_);
      CHECK_EQ(pool_output_.num(), num_);
      CHECK_EQ(pool_output_.channels(), channels_);
      CHECK_EQ(pool_output_.height(), height_);
      CHECK_EQ(pool_output_.width(), width_);
      // Set up power_layer_ to compute (1 + alpha_/N^2 s)^-beta_, where s is
      // the sum of a squared neighborhood (the output of pool_layer_).
      power_top_vec_.clear();
      power_top_vec_.push_back(&power_output_);
      LayerParameter power_param;
      power_param.mutable_power_param()->set_power(-beta_);
      power_param.mutable_power_param()->set_scale(alpha_);
      power_param.mutable_power_param()->set_shift(Dtype(1));
      power_layer_.reset(new PowerLayer<Dtype>(power_param));
      power_layer_->SetUp(pool_top_vec_, &power_top_vec_);
      CHECK_EQ(power_output_.num(), num_);
      CHECK_EQ(power_output_.channels(), channels_);
      CHECK_EQ(power_output_.height(), height_);
      CHECK_EQ(power_output_.width(), width_);
      // Set up a product_layer_ to compute outputs by multiplying inputs by the
      // inverse demoninator computed by the power layer.
      product_bottom_vec_.clear();
      product_bottom_vec_.push_back(bottom[0]);
      product_bottom_vec_.push_back(&power_output_);
      LayerParameter product_param;
      product_layer_.reset(new EltwiseProductLayer<Dtype>(product_param));
      product_layer_->SetUp(product_bottom_vec_, top);
      CHECK_EQ((*top)[0]->num(), num_);
      CHECK_EQ((*top)[0]->channels(), channels_);
      CHECK_EQ((*top)[0]->height(), height_);
      CHECK_EQ((*top)[0]->width(), width_);
    }
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
Dtype LRNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    return CrossChannelForward_cpu(bottom, top);
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    return WithinChannelForward(bottom, top);
  default:
    LOG(FATAL) << "Unknown normalization region.";
    return Dtype(0);
  }
}

template <typename Dtype>
Dtype LRNLayer<Dtype>::CrossChannelForward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // start with the constant value
  for (int i = 0; i < scale_.count(); ++i) {
    scale_data[i] = 1.;
  }
  Blob<Dtype> padded_square(1, channels_ + size_ - 1, height_, width_);
  Dtype* padded_square_data = padded_square.mutable_cpu_data();
  memset(padded_square_data, 0, sizeof(Dtype) * padded_square.count());
  Dtype alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the padded square
    caffe_sqr(channels_ * height_ * width_,
        bottom_data + bottom[0]->offset(n),
        padded_square_data + padded_square.offset(0, pre_pad_));
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c),
          scale_data + scale_.offset(n, 0));
    }
    for (int c = 1; c < channels_; ++c) {
      // copy previous scale
      caffe_copy<Dtype>(height_ * width_,
          scale_data + scale_.offset(n, c - 1),
          scale_data + scale_.offset(n, c));
      // add head
      caffe_axpy<Dtype>(height_ * width_, alpha_over_size,
          padded_square_data + padded_square.offset(0, c + size_ - 1),
          scale_data + scale_.offset(n, c));
      // subtract tail
      caffe_axpy<Dtype>(height_ * width_, -alpha_over_size,
          padded_square_data + padded_square.offset(0, c - 1),
          scale_data + scale_.offset(n, c));
    }
  }

  // In the end, compute output
  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data);
  caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);

  return Dtype(0.);
}

template <typename Dtype>
Dtype LRNLayer<Dtype>::WithinChannelForward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  split_layer_->Forward(bottom, &split_top_vec_);
  square_layer_->Forward(square_bottom_vec_, &square_top_vec_);
  pool_layer_->Forward(square_top_vec_, &pool_top_vec_);
  power_layer_->Forward(pool_top_vec_, &power_top_vec_);
  product_layer_->Forward(product_bottom_vec_, top);
  return Dtype(0.);
}

template <typename Dtype>
void LRNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  switch (this->layer_param_.lrn_param().norm_region()) {
  case LRNParameter_NormRegion_ACROSS_CHANNELS:
    CrossChannelBackward_cpu(top, propagate_down, bottom);
    break;
  case LRNParameter_NormRegion_WITHIN_CHANNEL:
    WithinChannelBackward(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown normalization region.";
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::CrossChannelBackward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = (*bottom)[0]->cpu_data();
  const Dtype* scale_data = scale_.cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Blob<Dtype> padded_ratio(1, channels_ + size_ - 1, height_, width_);
  Blob<Dtype> accum_ratio(1, 1, height_, width_);
  Dtype* padded_ratio_data = padded_ratio.mutable_cpu_data();
  Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
  // We hack a little bit by using the diff() to store an additional result
  Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
  memset(padded_ratio_data, 0, sizeof(Dtype) * padded_ratio.count());
  Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;

  caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
  caffe_mul<Dtype>(scale_.count(), top_diff, bottom_diff, bottom_diff);

  // go through individual data
  int inverse_pre_pad = size_ - (size_ + 1) / 2;
  for (int n = 0; n < num_; ++n) {
    int block_offset = scale_.offset(n);
    // first, compute diff_i * y_i / s_i
    caffe_mul<Dtype>(channels_ * height_ * width_,
        top_diff + block_offset, top_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    caffe_div<Dtype>(channels_ * height_ * width_,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad),
        scale_data + block_offset,
        padded_ratio_data + padded_ratio.offset(0, inverse_pre_pad));
    // Now, compute the accumulated ratios and the bottom diff
    memset(accum_ratio_data, 0, sizeof(Dtype) * accum_ratio.count());
    for (int c = 0; c < size_ - 1; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
    for (int c = 0; c < channels_; ++c) {
      caffe_axpy<Dtype>(height_ * width_, 1.,
          padded_ratio_data + padded_ratio.offset(0, c + size_ - 1),
          accum_ratio_data);
      // compute bottom diff
      caffe_mul<Dtype>(height_ * width_,
          bottom_data + top[0]->offset(n, c),
          accum_ratio_data, accum_ratio_times_bottom);
      caffe_axpy<Dtype>(height_ * width_, -cache_ratio_value,
          accum_ratio_times_bottom, bottom_diff + top[0]->offset(n, c));
      caffe_axpy<Dtype>(height_ * width_, -1.,
          padded_ratio_data + padded_ratio.offset(0, c), accum_ratio_data);
    }
  }
}

template <typename Dtype>
void LRNLayer<Dtype>::WithinChannelBackward(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    product_layer_->Backward(top, true, &product_bottom_vec_);
    power_layer_->Backward(power_top_vec_, true, &pool_top_vec_);
    pool_layer_->Backward(pool_top_vec_, true, &square_top_vec_);
    square_layer_->Backward(square_top_vec_, true, &square_bottom_vec_);
    split_layer_->Backward(split_top_vec_, true, bottom);
  }
}

INSTANTIATE_CLASS(LRNLayer);


}  // namespace caffe
