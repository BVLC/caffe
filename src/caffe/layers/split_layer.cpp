// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Split Layer takes a single blob as input.";
  CHECK_GE(top->size(), 1) << "Split Layer takes at least one blob as output.";
  count_ = bottom[0]->count();
  for (int i = 0; i < top->size(); ++i) {
    // Allow the 0th top blob to be 'in-place', but no others.
    if (i == 0 && (*top)[i] == bottom[0]) {
      continue;
    } else {
      CHECK_NE((*top)[i], bottom[0]) << "Only 0th top blob may be in place.";
    }
    (*top)[i]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                       bottom[0]->height(), bottom[0]->width());
    CHECK_EQ(count_, (*top)[i]->count());
  }
}

template <typename Dtype>
Dtype SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < top->size(); ++i) {
    (*top)[i]->ShareData(*bottom[0]);
  }
  return Dtype(0.);
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    (*bottom)[0]->ShareDiff(*top[0]);
    // Add remaining top blob diffs.
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    for (int i = 1; i < top.size(); ++i) {
      const Dtype* top_diff = top[i]->cpu_diff();
      caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
  }
}


INSTANTIATE_CLASS(SplitLayer);

}  // namespace caffe
