// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    (*top)[i]->ShareData(*bottom[0]);
  }
  return Dtype(0.);
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    (*bottom)[0]->ShareDiff(*top[0]);
    // Add remaining top blob diffs.
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    for (int i = 1; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->gpu_diff();
      caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
  }
}


INSTANTIATE_CLASS(SplitLayer);

}  // namespace caffe
