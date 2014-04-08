// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseProductLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_GE(bottom.size(), 2) <<
      "Eltwise Product Layer takes at least 2 blobs as input.";
  CHECK_EQ(top->size(), 1) <<
      "Eltwise Product Layer takes a single blob as output.";
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num, bottom[i]->num());
    CHECK_EQ(channels, bottom[i]->channels());
    CHECK_EQ(height, bottom[i]->height());
    CHECK_EQ(width, bottom[i]->width());
  }
  (*top)[0]->Reshape(num, channels, height, width);
}

template <typename Dtype>
Dtype EltwiseProductLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
  for (int i = 2; i < bottom.size(); ++i) {
    caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const int count = top[0]->count();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    for (int i = 0; i < bottom->size(); ++i) {
      const Dtype* bottom_data = (*bottom)[i]->cpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
      caffe_div(count, top_data, bottom_data, bottom_diff);
      caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_CLASS(EltwiseProductLayer);


}  // namespace caffe
