// Copyright 2013 Yangqing Jia

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
  KSIZE_ = this->layer_param_.kernelsize();
  STRIDE_ = this->layer_param_.stride();
  PAD_ = this->layer_param_.pad();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), CHANNELS_ * KSIZE_ * KSIZE_,
      (HEIGHT_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1,
      (WIDTH_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1);
}

template <typename Dtype>
void Im2colLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    im2col_cpu(bottom_data + bottom[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, top_data + (*top)[0]->offset(n));
  }
}

template <typename Dtype>
Dtype Im2colLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  for (int n = 0; n < top[0]->num(); ++n) {
    col2im_cpu(top_diff + top[0]->offset(n), CHANNELS_, HEIGHT_,
        WIDTH_, KSIZE_, PAD_, STRIDE_, bottom_diff + (*bottom)[0]->offset(n));
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(Im2colLayer);

}  // namespace caffe
