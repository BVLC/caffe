// Copyright 2014 Jeff Donahue

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
void SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int i = 0; i < top->size(); ++i) {
    if (i == 0 && (*top)[i] == bottom[0]) {
      continue;
    }
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    caffe_copy(count_, bottom_data, top_data);
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  for (int i = 0; i < top->size(); ++i) {
    if (i == 0 && (*top)[i] == bottom[0]) {
      continue;
    }
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    caffe_gpu_copy(count_, bottom_data, top_data);
  }
}

template <typename Dtype>
Dtype SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    // Initialize by copying first top blob diff to our diff, unless we're
    // doing in-place computation for the first blob, in which case the diff is
    // already initialized.
    if (top[0] != (*bottom)[0]) {
      caffe_copy(count_, top_diff, bottom_diff);
    }
    // Add remaining top blob diffs.
    for (int i = 1; i < top.size(); ++i) {
      top_diff = top[i]->cpu_diff();
      caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
  }
  return Dtype(0.);
}


template <typename Dtype>
Dtype SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    // Initialize by copying first top blob diff to our diff, unless we're
    // doing in-place computation for the first blob, in which case the diff is
    // already initialized.
    if (top[0] != (*bottom)[0]) {
      caffe_gpu_copy(count_, top_diff, bottom_diff);
    }
    // Add remaining top blob diffs.
    for (int i = 1; i < top.size(); ++i) {
      top_diff = top[i]->gpu_diff();
      caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
    }
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(SplitLayer);

}  // namespace caffe
