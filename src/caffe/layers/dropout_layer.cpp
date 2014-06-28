// Copyright 2014 BVLC and contributors.

// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::SetUp(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.reset(new Blob<unsigned int>(bottom[0]->num(),
      bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()));
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
Dtype DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->const_data();
  Dtype* top_data = (*top)[0]->mutable_data();
  int* mask = reinterpret_cast<int*>(rand_vec_->mutable_data());
  const int count = bottom[0]->count();
  if (Caffe::phase() == Caffe::TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * mask[i] * scale_;
    }
  } else {
    this->device_->copy(bottom[0]->count(), bottom_data, top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  CHECK(Caffe::phase() == Caffe::TRAIN);
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->const_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_diff();
    const unsigned int* mask = rand_vec_->const_data();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * mask[i] * scale_;
    }
  }
}


INSTANTIATE_CLASS(DropoutLayer);


}  // namespace caffe
