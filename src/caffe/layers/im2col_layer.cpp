// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void Im2colLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Im2col Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Im2col Layer takes a single blob as output.";
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  stride_ = this->layer_param_.convolution_param().stride();
  pad_ = this->layer_param_.convolution_param().pad();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_ * kernel_size_ * kernel_size_,
      (height_ + 2 * pad_ - kernel_size_) / stride_ + 1,
      (width_ + 2 * pad_ - kernel_size_) / stride_ + 1);
}

template <typename Dtype>
Dtype Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), channels_, height_,
        width_, kernel_size_, pad_, stride_, top_data + (*top)[0]->offset(n));
  }
  return Dtype(0.);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_cpu(top_diff + top[0]->offset(n), channels_, height_, width_,
        kernel_size_, pad_, stride_, bottom_diff + (*bottom)[0]->offset(n));
  }
}

INSTANTIATE_CLASS(Im2colLayer);

}  // namespace caffe
