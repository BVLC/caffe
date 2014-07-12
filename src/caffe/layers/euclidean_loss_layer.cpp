// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::FurtherSetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  this->device_->sub(
      count,
      bottom[0]->const_data(),
      bottom[1]->const_data(),
      diff_.mutable_data());
  Dtype dot = this->device_->dot(count, diff_.const_data(), diff_.const_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  if (top->size() == 1) {
    (*top)[0]->mutable_cpu_data()[0] = loss;
  }
  return loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      this->device_->axpby(
          (*bottom)[i]->count(),              // count
          sign / (*bottom)[i]->num(),         // alpha
          diff_.const_data(),                 // a
          Dtype(0),                           // beta
          (*bottom)[i]->mutable_diff());      // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);

}  // namespace caffe
