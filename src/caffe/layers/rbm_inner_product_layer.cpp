#include <algorithm>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  setup_sizes_.clear();
  setup_sizes_.push_back(bottom.size());
  setup_sizes_.push_back(top.size());
  bool skip_init = (this->blobs_.size() > 0);
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();
  num_error_                   = param.loss_measure_size();
  visible_bias_term_           = param.visible_bias_term();
  num_sample_steps_for_update_ = param.sample_steps_in_update();
  forward_is_update_           = param.forward_is_update();
  vector<int> starting_bottom_shape = bottom[0]->shape();
  const int axis = 1;
  num_visible_ = bottom[0]->count(axis);
  batch_size_ = bottom[0]->count(0, axis);
  CHECK_GE(top.size(), num_error_)
      << "top must be at least as large as the number of errors";
  CHECK_LE(top.size(), num_error_ + 3)
      << "top must be no larger than number error plus 3";
  // set up the hidden pre_activation_layer_ vector
  pre_activation_h1_vec_.clear();
  if (top.size() >= num_error_ + 1) {
    pre_activation_h1_vec_.push_back(top[0]);
  } else {
    pre_activation_h1_blob_.reset(new Blob<Dtype>());
    pre_activation_h1_vec_.push_back(pre_activation_h1_blob_.get());
  }
  sample_v1_vec_.clear();
  sample_v1_vec_.push_back(bottom[0]);
  if (!skip_init) {
    // set up the connection layer
    CHECK(param.has_connection_layer_param())
      << "a connection layer must be specified for the RBMInnerProductLayer";
    connection_layer_ =
      LayerRegistry<Dtype>::CreateLayer(param.connection_layer_param());
  }
  connection_layer_->SetUp(sample_v1_vec_, pre_activation_h1_vec_);

  // set up the hidden post_activation_layer_ vector
  post_activation_h1_vec_.clear();
  if (top.size() >= num_error_ + 2) {
    post_activation_h1_vec_.push_back(top[1]);
  } else {
    post_activation_h1_blob_.reset(new Blob<Dtype>());
    post_activation_h1_vec_.push_back(post_activation_h1_blob_.get());
  }

  // set up the hidden_activation_layer
  if (param.has_hidden_activation_layer_param()) {
    if (!skip_init) {
      hidden_activation_layer_ =
      LayerRegistry<Dtype>::CreateLayer(param.hidden_activation_layer_param());
    }
    hidden_activation_layer_->SetUp(pre_activation_h1_vec_,
                                    post_activation_h1_vec_);
  } else {
    pre_activation_h1_vec_[0]->ShareData(*post_activation_h1_vec_[0]);
  }

  // set up the hidden sample vector
  sample_h1_vec_.clear();
  if (top.size() >= num_error_ + 3) {
    sample_h1_vec_.push_back(top[2]);
  } else {
    sample_h1_blob_.reset(new Blob<Dtype>());
    sample_h1_vec_.push_back(sample_h1_blob_.get());
  }

  // set up the hidden_sampling_layer_
  if (param.has_hidden_sampling_layer_param()) {
    if (!skip_init) {
      hidden_sampling_layer_ =
        LayerRegistry<Dtype>::CreateLayer(param.hidden_sampling_layer_param());
    }
    hidden_sampling_layer_->SetUp(post_activation_h1_vec_,
                                  sample_h1_vec_);
  } else {
    post_activation_h1_vec_[0]->ShareData(*sample_h1_vec_[0]);
  }

  // set up the visible pre and post_activation vectors
  pre_activation_v1_blob_.reset(new Blob<Dtype>(bottom[0]->shape()));
  pre_activation_v1_blob_->ShareDiff(*bottom[0]);
  pre_activation_v1_vec_.clear();
  pre_activation_v1_vec_.push_back(pre_activation_v1_blob_.get());
  post_activation_v1_blob_.reset(new Blob<Dtype>(bottom[0]->shape()));
  post_activation_v1_vec_.clear();
  post_activation_v1_vec_.push_back(post_activation_v1_blob_.get());

  // set up the visible_activation_layer
  if (param.has_visible_activation_layer_param()) {
    if (!skip_init) {
      visible_activation_layer_ =
      LayerRegistry<Dtype>::CreateLayer(param.visible_activation_layer_param());
    }
    visible_activation_layer_->SetUp(pre_activation_v1_vec_,
                                     post_activation_v1_vec_);
  } else {
    post_activation_v1_vec_[0]->ShareData(*pre_activation_v1_vec_[0]);
  }

  // set up the visible_sampling_layer_
  if (param.has_visible_sampling_layer_param()) {
    if (!skip_init) {
      visible_sampling_layer_ =
        LayerRegistry<Dtype>::CreateLayer(param.visible_sampling_layer_param());
    }
    visible_sampling_layer_->SetUp(post_activation_v1_vec_,
                                   sample_v1_vec_);
  } else {
    post_activation_v1_vec_[0]->ShareData(*sample_v1_vec_[0]);
  }

  CHECK_EQ(starting_bottom_shape.size(), bottom[0]->shape().size())
      << "something is wrong, bottom should not change size";
  for (int i = 0; i < starting_bottom_shape.size(); ++i) {
    CHECK_EQ(starting_bottom_shape[i], bottom[0]->shape()[i])
        << "something is wrong, bottom should not change size";
  }

  bottom_copy_.reset(new Blob<Dtype>(bottom[0]->shape()));

  // Set up the visible bias.
  if (skip_init) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Add the blobs from the connection layer to the rbm blobs:
    vector<shared_ptr<Blob<Dtype> > >& connection_blobs =
        connection_layer_->blobs();
    this->blobs_.resize(connection_blobs.size());
    for (int i = 0; i < connection_blobs.size(); ++i) {
      this->blobs_[i] = connection_blobs[i];
    }
    // Add the blob for the visible bias if required.
    if (visible_bias_term_) {
      visible_bias_index_ = this->blobs_.size();
      vector<int> bias_shape(pre_activation_v1_blob_->shape());
      bias_shape[0] = 1;
      this->blobs_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>(bias_shape)));
      // Fill the visible bias.
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.rbm_inner_product_param().visible_bias_filler()));
      bias_filler->Fill(this->blobs_[visible_bias_index_].get());
    }
  }
}

template <typename Dtype>
void reshape_error(int index, int num_error, int batch_size,
    const RBMInnerProductParameter& param, const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int top_index = top.size() + index - num_error;
  vector<int> blob_shape(1, batch_size);
  switch (param.loss_measure(index)) {
  case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
    top[top_index]->ReshapeLike(*bottom[0]);
    break;
  case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
    top[top_index]->Reshape(blob_shape);
    break;
  default:
      LOG(FATAL) << "Unknown loss measure: " << param.loss_measure(index);
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), setup_sizes_[0])
      << "bottom must be the same size at SetUp, Reshape and Forward";
  CHECK_EQ(top.size(), setup_sizes_[1])
      << "top must be the same size at SetUp, Reshape and Forward";
  // Figure out the dimensions
  vector<int> starting_bottom_shape = bottom[0]->shape();
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();

  // better safe than sorry
  sample_v1_vec_[0] = bottom[0];

  if (top.size() >= num_error_ + 1) {
    pre_activation_h1_vec_[0] = top[0];
  } else {
    pre_activation_h1_vec_[0] = pre_activation_h1_blob_.get();
  }

  if (top.size() >= num_error_ + 2) {
    post_activation_h1_vec_[0] = top[1];
  } else {
    post_activation_h1_vec_[0] = post_activation_h1_blob_.get();
  }

  if (top.size() >= num_error_ + 3) {
    sample_h1_vec_[0] = top[2];
  } else {
    sample_h1_vec_[0] = sample_h1_blob_.get();
  }

  // Reshape each of the layers for the forward pass.
  connection_layer_->Reshape(sample_v1_vec_, pre_activation_h1_vec_);
  if (param.has_hidden_activation_layer_param()) {
    hidden_activation_layer_->Reshape(pre_activation_h1_vec_,
                                      post_activation_h1_vec_);
  }
  if (param.has_hidden_sampling_layer_param()) {
    hidden_sampling_layer_->Reshape(post_activation_h1_vec_, sample_h1_vec_);
  }

  // The blobs for hidden to visible steps must also be reshaped.
  pre_activation_v1_vec_[0]->ReshapeLike(*bottom[0]);
  if (param.has_visible_activation_layer_param()) {
    visible_activation_layer_->Reshape(pre_activation_v1_vec_,
                                       post_activation_v1_vec_);
  }
  if (param.has_visible_sampling_layer_param()) {
    visible_sampling_layer_->Reshape(post_activation_v1_vec_, sample_v1_vec_);
  }

  CHECK_EQ(starting_bottom_shape.size(), bottom[0]->shape().size())
      << "something is wrong, bottom should not change size";
  for (int i = 0; i < starting_bottom_shape.size(); ++i) {
    CHECK_EQ(starting_bottom_shape[i], bottom[0]->shape()[i])
        << "something is wrong, bottom should not change size";
  }

  int max_count = std::max(sample_v1_vec_[0]->count(1),
                           pre_activation_h1_vec_[0]->shape(0));
  if (visible_bias_term_ && (bias_multiplier_.num_axes() < 2 ||
                             bias_multiplier_.count() < max_count)) {
    // we use this bias_multiplier_ to multiply both the hidden and vis bias
    vector<int> bias_multiplier_shape(1, max_count);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(max_count, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  for (int i = 0; i < num_error_; ++i) {
    reshape_error(i, num_error_, batch_size_, param, bottom, top);
  }

  bottom_copy_->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(RBMInnerProductLayer);
REGISTER_LAYER_CLASS(RBMInnerProduct);

}  // namespace caffe
