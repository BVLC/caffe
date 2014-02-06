// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FlattenLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Flatten Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Flatten Layer takes a single blob as output.";
  int channels_out = bottom[0]->channels() * bottom[0]->height()
      * bottom[0]->width();
  (*top)[0]->Reshape(bottom[0]->num(), channels_out, 1, 1);
  count_ = bottom[0]->num() * channels_out;
  CHECK_EQ(count_, bottom[0]->count());
  CHECK_EQ(count_, (*top)[0]->count());
};

template <typename Dtype>
void FlattenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_copy(count_, bottom_data, top_data);
}

template <typename Dtype>
void FlattenLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_copy(count_, bottom_data, top_data);
}

template <typename Dtype>
Dtype FlattenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_copy(count_, top_diff, bottom_diff);
  return Dtype(0.);
}


template <typename Dtype>
Dtype FlattenLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  caffe_gpu_copy(count_, top_diff, bottom_diff);
  return Dtype(0.);
}

INSTANTIATE_CLASS(FlattenLayer);

}  // namespace caffe
