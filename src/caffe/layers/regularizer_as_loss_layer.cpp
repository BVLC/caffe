// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::vector;

template<typename Dtype>
RegularizerAsLossLayer<Dtype>::RegularizerAsLossLayer(
    const LayerParameter& param)
    : Layer<Dtype>(param),
      num_regularizers_(param.regularizer_size()) {
  if (num_regularizers_ > 0) {
    regularizers_.resize(num_regularizers_);
    for (int i = 0; i < num_regularizers_; ++i) {
      regularizers_[i].reset(GetRegularizer<Dtype>(param.regularizer(i)));
    }
  }
}

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1)<<
      "RegularizerAsLossLayer takes one blob as input.";
  CHECK_EQ(top->size(), 0) <<
      "RegularizerAsLossLayer takes no blob as output.";
}

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Blob<Dtype>* bottom_ptr = bottom[0];
  if (bottom_ptr->count() <= 0) {
  } else {
    memset(bottom_ptr->mutable_cpu_diff(), 0,
           bottom_ptr->count() * sizeof(Dtype));
    Dtype loss = 0;
    for (int i = 0; i < num_regularizers_; ++i) {
      loss += regularizers_[i]->Regularize_cpu(bottom_ptr);
    }
    int num = bottom_ptr->num();
    // Scale down gradient
    caffe_scal<Dtype>(bottom_ptr->count(), Dtype(1) / num,
                      bottom_ptr->mutable_cpu_diff());
    return loss / num;
  }
}

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  return;
}

INSTANTIATE_CLASS(RegularizerAsLossLayer);

}  // namespace caffe
