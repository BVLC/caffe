// Copyright 2014 Sergio Guadarama

#include <vector>
#include <string>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Reshape Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Reshape Layer takes a single blob as output.";
  NUM_ = this->layer_param_.num();
  CHANNEL_ =  this->layer_param_.channels();
  HEIGHT_ = this->layer_param_.height();
  WIDTH_ = this->layer_param_.width();
  count_ = NUM_*CHANNEL_*HEIGHT_*WIDTH_;
  (*top)[0]->Reshape(NUM_, CHANNEL_, HEIGHT_, WIDTH_);  
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, (*top)[0]->count());
};

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_copy(count_, bottom_data, top_data);
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_copy(count_, bottom_data, top_data);
}

template <typename Dtype>
Dtype ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_copy(count_, top_diff, bottom_diff);
  return Dtype(0.);
}


template <typename Dtype>
Dtype ReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  caffe_gpu_copy(count_, top_diff, bottom_diff);
  return Dtype(0.);
}

INSTANTIATE_CLASS(ReshapeLayer);

}  // namespace caffe
