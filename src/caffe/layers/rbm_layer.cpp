// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void RBMLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                            vector<Blob<Dtype>*>* top) {
  CHECK(this->layer_param_.has_rbm_param());
  CHECK(this->layer_param_.rbm_param().has_hidden_dim());
  CHECK(this->layer_param_.rbm_param().has_weight_filler());
  hidden_dim_ = this->layer_param_.rbm_param().hidden_dim();
  CHECK_EQ(bottom.size(), 1) << "RBM Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "RBM Layer takes a single blob as output.";
  visible_dim_ = bottom[0]->count() / bottom[0]->num();
  (*top)[0]->Reshape(bottom[0]->num(), hidden_dim_, 1, 1);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO)<< "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    visible_hidden_weight_ = this->blobs_[0];
    visible_bias_ = this->blobs_[1];
    hidden_bias_ = this->blobs_[2];
    // Intialize the visible-hidden weight
    visible_hidden_weight_.reset(new Blob<Dtype>(1, 1, hidden_dim_,
            visible_dim_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.rbm_param().weight_filler()));
    weight_filler->Fill(visible_hidden_weight_.get());
    visible_bias_.reset(new Blob<Dtype>(1, 1, 1, visible_dim_));
    memset(this->blobs_[1]->mutable_cpu_data(), 0,
        sizeof(Dtype) * visible_bias_->count());
    hidden_bias_.reset(new Blob<Dtype>(1, 1, 1, hidden_dim_));
    Dtype* hidden_bias_data = hidden_bias_->mutable_cpu_data();
    for (int i = 0; i < hidden_bias_->count(); ++i) {
      hidden_bias_data[i] = -4;
    }
  }  // parameter initialization
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  hidden_unit_sampling_filler_.reset(new UniformFiller<Dtype>(filler_param));

  int num = bottom[0]->num();
  pos_hidden_activations_.reset(
      new Blob<Dtype>(num, hidden_dim_, 1, 1));
  pos_hidden_probs_.reset(
        new Blob<Dtype>(num, hidden_dim_, 1, 1));
  pos_hidden_states_.reset(
      new Blob<Dtype>(num, hidden_dim_, 1, 1));
  pos_association_.reset(
      new Blob<Dtype>(visible_dim_, hidden_dim_, 1, 1));
  random_threshold_.reset(
      new Blob<Dtype>(num, hidden_dim_, 1, 1));
  neg_visible_activations_.reset(
        new Blob<Dtype>(num, visible_dim_, 1, 1));
  neg_visible_probs_.reset(
        new Blob<Dtype>(num, visible_dim_, 1, 1));
  neg_hidden_activations_.reset(
        new Blob<Dtype>(num, visible_dim_, 1, 1));
  neg_hidden_probs_.reset(
        new Blob<Dtype>(num, visible_dim_, 1, 1));
  neg_associations_.reset(
      new Blob<Dtype>(num, visible_dim_, 1, 1));
}

template<typename Dtype>
Dtype RBMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* pos_hidden_activations_data = pos_hidden_activations_->mutable_cpu_data();
  const Dtype* visible_hidden_weight_data = visible_hidden_weight_->cpu_data();
  int num = bottom[0]->num();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, hidden_dim_,
                        visible_dim_, (Dtype) 1., bottom_data,
                        visible_hidden_weight_data, (Dtype) 0.,
                        pos_hidden_activations_data);
  const Dtype* hidden_bias_data = hidden_bias_->cpu_data();
  for (int i = 0; i < num; ++i) {
    caffe_axpy<Dtype>(
        hidden_dim_, 1, hidden_bias_data,
        pos_hidden_activations_data + pos_hidden_activations_->offset(i));
  }
  Dtype* pos_hidden_probs_data = pos_hidden_probs_->mutable_cpu_data();
  caffe_cpu_sigmoid<Dtype>(pos_hidden_activations_->count(),
                    pos_hidden_activations_data, pos_hidden_probs_data);
  // sampling hidden units
  hidden_unit_sampling_filler_->Fill(random_threshold_.get());
  const Dtype* random_threshold_data = random_threshold_->cpu_data();
  Dtype* random_threshold_diff = random_threshold_->mutable_cpu_diff();
  Dtype* pos_hidden_states_data = pos_hidden_states_->mutable_cpu_data();
  const int count = pos_hidden_states_->count();
  caffe_sub<Dtype>(count, random_threshold_data, pos_hidden_states_data,
            random_threshold_diff);
  caffe_cpu_sgnbit<Dtype>(count, random_threshold_diff, pos_hidden_states_data);
  Dtype* pos_association_data = pos_association_->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, visible_dim_, hidden_dim_,
                        num, (Dtype) 1., bottom_data, (*top)[0]->cpu_data(),
                        (Dtype) 0., pos_association_data);
  return Dtype(0);
}
//# Clamp to the data and sample from the hidden units.
//     # (This is the "positive CD phase", aka the reality phase.)
//     pos_hidden_activations = np.dot(data, self.weights)
//     pos_hidden_probs = self._logistic(pos_hidden_activations)
//     pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
//     # Note that we're using the activation *probabilities* of the hidden states, not the hidden states
//     # themselves, when computing associations. We could also use the states; see section 3 of Hinton's
//     # "A Practical Guide to Training Restricted Boltzmann Machines" for more.
//     pos_associations = np.dot(data.T, pos_hidden_probs)
//
//     # Reconstruct the visible units and sample again from the hidden units.
//     # (This is the "negative CD phase", aka the daydreaming phase.)
//     neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
//     neg_visible_probs = self._logistic(neg_visible_activations)
//     neg_visible_probs[:,0] = 1 # Fix the bias unit.
//     neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
//     neg_hidden_probs = self._logistic(neg_hidden_activations)
//     # Note, again, that we're using the activation *probabilities* when computing associations, not the states
//     # themselves.
//     neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

template<typename Dtype>
void RBMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const bool propagate_down,
                                   vector<Blob<Dtype>*>* bottom) {
  int num = top[0]->num();
  const Dtype* pos_hidden_states_data = pos_hidden_states_->cpu_diff();
  Dtype* neg_visible_activations_data =
      neg_visible_activations_->mutable_cpu_data();
  const Dtype* visible_hidden_weight_data = visible_hidden_weight_->cpu_data();
  caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasTrans, num, hidden_dim_, visible_dim_, (Dtype) 1.,
      pos_hidden_states_data, visible_hidden_weight_data, (Dtype) 0.,
      neg_visible_activations_data);
  Dtype* neg_visible_probs_data = neg_visible_probs_->mutable_cpu_data();
  caffe_cpu_sigmoid<Dtype>(neg_visible_activations_->count(),
                    neg_visible_activations_data,
                    neg_visible_probs_data);
  Dtype* neg_hidden_activations_data =
      neg_hidden_activations_->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(
      CblasNoTrans, CblasNoTrans, num, hidden_dim_, visible_dim_, (Dtype) 1.,
      neg_visible_probs_data, visible_hidden_weight_data, (Dtype) 0.,
      neg_hidden_activations_data);
  Dtype* neg_hidden_probs_data = neg_hidden_probs_->mutable_cpu_data();
  caffe_cpu_sigmoid<Dtype>(neg_hidden_activations_->count(),
                    neg_hidden_activations_data,
                    neg_hidden_probs_data);
  Dtype* neg_associations_data = neg_associations_->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(
      CblasTrans, CblasNoTrans, visible_dim_, hidden_dim_, num, (Dtype) 1.,
      neg_visible_probs_data, neg_hidden_probs_data, (Dtype) 0.,
      neg_associations_data);
}

INSTANTIATE_CLASS(RBMLayer);

}  // namespace caffe
