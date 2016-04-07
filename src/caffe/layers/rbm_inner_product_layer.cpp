#include <algorithm>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/rbm_inner_product_layer.hpp"
#include "caffe/util/device_alternate.hpp"

namespace caffe {

/** @brief The following class provides a wrapper for a Blob which swaps the
 * data_ and diff_ arrays.
 **/
template <typename Dtype>
class SwapBlob: public Blob<Dtype> {
 public:
  /// @brief shallow copy that copies and swaps smart pointers to data and diff
  explicit SwapBlob(const Blob<Dtype>* other) : Blob<Dtype>() {
    SetUp(other);
  }
  void SetUp(const Blob<Dtype>* other) {
    this->diff_ = other->data();
    this->data_ = other->diff();
    this->capacity_ = other->count();
    this->count_ = other->count();
    // since capacity was set, this resize does not reallocate
    this->ReshapeLike(*other);
  }
 private:
  // disable default construct and assign
  explicit SwapBlob() {}
};

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
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();

  if (param.forward_is_update()) {
    bottom_copy_->CopyFrom(*bottom[0]);
    vector<Blob<Dtype>*> bottom_samples_vector(1);
    bottom_samples_vector[0] = bottom_copy_.get();
    // create a top with all three processing steps
    vector<Blob<Dtype>*> full_top;
    full_top.push_back(this->pre_activation_h1_vec_[0]);
    full_top.push_back(this->post_activation_h1_vec_[0]);
    full_top.push_back(this->sample_h1_vec_[0]);

    for (int i = 0; i < num_error_; ++i) {
      full_top.push_back(top[top.size() + i - num_error_]);
    }

    // sample forward
    sample_h_given_v(bottom_samples_vector, full_top);

    // do some sampling and then an update
    gibbs_hvh(bottom_samples_vector, full_top);
  } else {
    // just sample forwards
    sample_h_given_v(bottom, top);
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    // Disable the update of the diffs for the weights.
    for (int i = 0; i < connection_layer_->blobs().size(); ++i) {
      connection_layer_->set_param_propagate_down(i, false);
    }
    vector<Blob<Dtype>*> hidden(0);
    SwapBlob<Dtype> hidden_blob(top[0]);
    hidden.push_back(&hidden_blob);
    sample_v_given_h(bottom, hidden);

    // Enable the update of the diffs for the weights.
    for (int i = 0; i < connection_layer_->blobs().size(); ++i) {
      connection_layer_->set_param_propagate_down(i, true);
    }
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::sample_h_given_v(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();
  // Do a forward pass through each of the layers.
  if (top.size() >= num_error_ + 1) {
    connection_layer_->Forward(bottom, pre_activation_h1_vec_);
    if (top.size() >= num_error_ + 2) {
      if (param.has_hidden_activation_layer_param()) {
        hidden_activation_layer_->Forward(pre_activation_h1_vec_,
                                          post_activation_h1_vec_);
      }
      if (top.size() >= num_error_ + 3) {
        if (param.has_hidden_sampling_layer_param()) {
          hidden_sampling_layer_->Forward(post_activation_h1_vec_,
                                          sample_h1_vec_);
        }
      }
    }
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::update_diffs(const int k,
    const vector<Blob<Dtype>*>& hidden_k,
    const vector<Blob<Dtype>*>& visible_k) {

  SwapBlob<Dtype> swapped_hidden_k(hidden_k[1]);
  vector<Blob<Dtype>*> hidden_vec;

  // Update the diffs for the weights and hidden bias
  vector<bool> propagate_down(1, false);

  Blob<Dtype> scaled_k;
  if (k == 0) {
    // In order to get the summation to work out correctly, we need to multiply
    // the hidden vector by -1.
    scaled_k.CopyFrom(*hidden_k[1], false, true);
    scaled_k.scale_data((Dtype)-1.);
    swapped_hidden_k.SetUp(&scaled_k);
  }
  hidden_vec.push_back(&swapped_hidden_k);
  connection_layer_->Backward(hidden_vec, propagate_down, visible_k);

  // Update the diffs for the visible bias
  // Update the visible bias diff (delta b -= v_k).
  if (visible_bias_term_) {
    Dtype* vbias_diff = 0;  // init to supress compiler warnings
    const Dtype factor = (k != 0) ? (Dtype)(1) : (Dtype)(-1.);
    switch (Caffe::mode()) {
    case Caffe::CPU:
      vbias_diff = this->blobs_[visible_bias_index_]->mutable_cpu_diff();
      caffe_cpu_gemv<Dtype>(CblasTrans, batch_size_, num_visible_, factor,
                            visible_k[0]->cpu_data(),
                            bias_multiplier_.cpu_data(),
                            (Dtype)1., vbias_diff);
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      vbias_diff = this->blobs_[visible_bias_index_]->mutable_gpu_diff();
      caffe_gpu_gemv<Dtype>(CblasTrans, batch_size_, num_visible_, factor,
                            visible_k[0]->gpu_data(),
                            bias_multiplier_.gpu_data(),
                            (Dtype)1., vbias_diff);
#else
      NO_GPU;
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::gibbs_hvh(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();

  // Update the diffs for k = 0, P(h|v_0)
  update_diffs(0, top, bottom);

  // Disable the update of the diffs for the weights.
  for (int i = 0; i < connection_layer_->blobs().size(); ++i) {
    connection_layer_->set_param_propagate_down(i, false);
  }

  // Perform k Gibbs sampling steps.
  for (int k = 0; k < num_sample_steps_for_update_; k++) {
    // Down propagation
    sample_v_given_h(bottom, top);

    // copy reconstruction error to top
    if (k == 0) {
      for (int i = 0; i < num_error_; ++i) {
        int top_index = top.size() + i - num_error_;
        switch (param.loss_measure(i)) {
          case RBMInnerProductParameter_LossMeasure_RECONSTRUCTION:
            top[top_index]->CopyFrom(*bottom[0]);
            break;
          case RBMInnerProductParameter_LossMeasure_FREE_ENERGY:
            LOG(FATAL) << "FREE_ENERGY not implemented yet ";
          default:
            LOG(FATAL) << "Unknown loss measure: " << param.loss_measure(i);
        }
      }
    }

    // Up propagation
    sample_h_given_v(bottom, top);
  }

  // Enable the update of the diffs for the weights and hidden bias again.
  for (int i = 0; i < connection_layer_->blobs().size(); ++i) {
    connection_layer_->set_param_propagate_down(i, true);
  }

  // Update the diffs for k, P(h|v_k)
  update_diffs(num_sample_steps_for_update_, top, bottom);
}

template <typename Dtype>
void RBMInnerProductLayer<Dtype>::sample_v_given_h(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const RBMInnerProductParameter& param =
      this->layer_param_.rbm_inner_product_param();
  SwapBlob<Dtype> swapped_top(top[0]);
  vector<Blob<Dtype>*> h1;
  h1.push_back(&swapped_top);

  // Backward pass through the connection layer, save to pre_activation diffs
  vector<bool> propagate_down(1, true);
  connection_layer_->Backward(h1, propagate_down, pre_activation_v1_vec_);
  Dtype one = 1;
  // Add the visible bias to the pre activation.
  if (visible_bias_term_) {
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->batch_size_,
                            this->num_visible_, 1, one,
                            this->bias_multiplier_.cpu_data(),
                            this->blobs_[visible_bias_index_]->cpu_data(), one,
                            pre_activation_v1_vec_[0]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->batch_size_,
                            this->num_visible_, 1, one,
                            this->bias_multiplier_.gpu_data(),
                            this->blobs_[visible_bias_index_]->gpu_data(), one,
                            pre_activation_v1_vec_[0]->mutable_gpu_diff());
#else
      NO_GPU;
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
  }

  // swap diff and data for pre_activation to do sqash and sample
  SwapBlob<Dtype> swapped_pre_activation(pre_activation_v1_vec_[0]);
  vector<Blob<Dtype>*> pre_activation_v1;
  pre_activation_v1.push_back(&swapped_pre_activation);

  // Do a forward pass through the activation layer.
  if (param.has_visible_activation_layer_param()) {
    visible_activation_layer_->Forward(pre_activation_v1,
                                       post_activation_v1_vec_);
  }

  // Sample the mean field and store this in the bottom.
  if (param.has_visible_sampling_layer_param()) {
    visible_sampling_layer_->Forward(post_activation_v1_vec_, bottom);
  }
}

INSTANTIATE_CLASS(RBMInnerProductLayer);
REGISTER_LAYER_CLASS(RBMInnerProduct);

}  // namespace caffe
