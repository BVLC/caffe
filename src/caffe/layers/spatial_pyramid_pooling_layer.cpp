// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::SetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  num_pyramid_levels_ =
      this->layer_param_.spatial_pyramid_pooling_param().spatial_bin_size();
  CHECK_GE(num_pyramid_levels_, 1);
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // TODO: Support unequal height and width sizes once PoolingLayer does
  CHECK_GE(height_, width_);
  const int image_side_length = height_;
  // TODO: multiple image scales
  // float scale = this->layer_param_.spatial_pyramid_pooling_param().scale();

  LayerParameter layer_param;
  layer_param.mutable_concat_param()->set_concat_dim(1);
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  switch (this->layer_param_.pooling_param().pool()) {
  case SpatialPyramidPoolingParameter_PoolMethod_MAX:
    pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
    break;
  case SpatialPyramidPoolingParameter_PoolMethod_AVE:
    pooling_param->set_pool(PoolingParameter_PoolMethod_AVE);
    break;
  case SpatialPyramidPoolingParameter_PoolMethod_STOCHASTIC:
    pooling_param->set_pool(PoolingParameter_PoolMethod_STOCHASTIC);
    break;
  default:
    LOG(FATAL) << "Unknown spatial pyramid pooling method " <<
      this->layer_param_.pooling_param().pool();
  }

  if (num_pyramid_levels_ > 1) {
    split_top_vec_.clear();
    split_top_vec_.push_back(bottom[0]);
    for (int i = 1; i < num_pyramid_levels_; ++i) {
      split_top_vec_.push_back(new Blob<Dtype>());
    }
    split_layer_.reset(new SplitLayer<Dtype>(layer_param));
    split_layer_->SetUp(bottom, &split_top_vec_);
    pooling_bottom_vecs_.clear();
    pooling_top_vecs_.clear();
    pooling_layers_.clear();
    flatten_top_vecs_.clear();
    flatten_layers_.clear();
    concat_bottom_vec_.clear();
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      const int spatial_bin =
          this->layer_param_.spatial_pyramid_pooling_param().spatial_bin(i);
      const float spatial_bin_size =
          static_cast<float>(image_side_length) / spatial_bin;
      pooling_param->set_kernel_size(ceil(spatial_bin_size));
      pooling_param->set_stride(floor(spatial_bin_size));
      shared_ptr<PoolingLayer<Dtype> > pooling_layer(
          new PoolingLayer<Dtype>(layer_param));
      vector<Blob<Dtype>*> pooling_layer_bottom(1, split_top_vec_[i]);
      vector<Blob<Dtype>*> pooling_layer_top(1, new Blob<Dtype>());
      pooling_layer->SetUp(pooling_layer_bottom, &pooling_layer_top);
      pooling_bottom_vecs_.push_back(pooling_layer_bottom);
      pooling_top_vecs_.push_back(pooling_layer_top);
      pooling_layers_.push_back(pooling_layer);

      shared_ptr<FlattenLayer<Dtype> > flatten_layer(
          new FlattenLayer<Dtype>(layer_param));
      vector<Blob<Dtype>*> flatten_layer_top(1, new Blob<Dtype>());
      flatten_layer->SetUp(pooling_layer_top, &flatten_layer_top);
      flatten_top_vecs_.push_back(flatten_layer_top);
      flatten_layers_.push_back(flatten_layer);

      concat_bottom_vec_.push_back(flatten_layer_top[0]);
    }
    concat_layer_.reset(new ConcatLayer<Dtype>(layer_param));
    concat_layer_->SetUp(concat_bottom_vec_, top);
  } else {
    pooling_param->set_kernel_size(image_side_length);
    pooling_param->set_stride(image_side_length);
    shared_ptr<PoolingLayer<Dtype> > layer(
        new PoolingLayer<Dtype>(layer_param));
    layer->SetUp(bottom, top);
    pooling_layers_.push_back(layer);
  }
}

template <typename Dtype>
Dtype SpatialPyramidPoolingLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype loss = 0;
  if (num_pyramid_levels_ > 1) {
    split_layer_->Forward(bottom, &split_top_vec_);
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      loss += pooling_layers_[i]->Forward(pooling_bottom_vecs_[i],
                                          &(pooling_top_vecs_[i]));
      loss += flatten_layers_[i]->Forward(pooling_top_vecs_[i],
                                          &(flatten_top_vecs_[i]));
    }
    loss += concat_layer_->Forward(concat_bottom_vec_, top);
  } else {
    loss = pooling_layers_[0]->Forward(bottom, top);
  }
  return Dtype(0);
}

template <typename Dtype>
void SpatialPyramidPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  if (num_pyramid_levels_ > 1) {
    concat_layer_->Backward(top, propagate_down, &concat_bottom_vec_);
    for (int i = 0; i < num_pyramid_levels_; ++i) {
      flatten_layers_[i]->Backward(flatten_top_vecs_[i], propagate_down,
                                   &(pooling_top_vecs_[i]));
      pooling_layers_[i]->Backward(pooling_top_vecs_[i], propagate_down,
                                   &(pooling_bottom_vecs_[i]));
    }
    split_layer_->Backward(split_top_vec_, propagate_down, bottom);
  } else {
    pooling_layers_[0]->Backward(top, propagate_down, bottom);
  }
}


INSTANTIATE_CLASS(SpatialPyramidPoolingLayer);


}  // namespace caffe
