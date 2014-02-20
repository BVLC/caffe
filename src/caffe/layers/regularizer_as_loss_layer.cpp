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
    : Layer<Dtype>(param) {
  num_regularizers_ = param.regularizer_size();
  for (int s = 0; s < num_regularizers_; ++s) {
    regularizers_.push_back(
        shared_ptr<Regularizer<Dtype> >(
            GetRegularizer<Dtype>(param.regularizer(s))));
  }
}

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                          vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1)<< "RegularizerAsLossLayer takes one blob as input.";
  CHECK_EQ(top->size(), 0) << "RegularizerAsLossLayer takes no blob as output.";
};

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
}

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  Blob<Dtype>* bottom_ptr = bottom->at(0);
  if (bottom_ptr->count() <= 0) {
    return Dtype(0);
  } else {
    memset(bottom_ptr->mutable_cpu_diff(), 0,
           bottom_ptr->count() * sizeof(Dtype));
    Dtype loss = 0;
    for (int s = 0; s < num_regularizers_; ++s) {
      loss += regularizers_[s]->Regularize_cpu(bottom_ptr);
    }
    int num = bottom_ptr->num();
    // Scale down gradient
    caffe_scal<Dtype>(bottom_ptr->count(), Dtype(1) / num,
                      bottom_ptr->mutable_cpu_diff());
    return loss / num;
  }
}

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  Blob<Dtype>* bottom_ptr = bottom->at(0);
  if (bottom_ptr->count() <= 0) {
    return Dtype(0);
  } else {
    CUDA_CHECK(
        cudaMemset(bottom_ptr->mutable_gpu_diff(), 0,
                   bottom_ptr->count() * sizeof(Dtype)));
    Dtype loss = 0;
    for (int s = 0; s < num_regularizers_; ++s) {
      loss += regularizers_[s]->Regularize_cpu(bottom_ptr);
    }
    int num = bottom_ptr->num();
    // Scale down gradient
    caffe_gpu_scal<Dtype>(bottom_ptr->count(), Dtype(1) / num,
                          bottom_ptr->mutable_gpu_diff());
    return loss / num;
  }
}

INSTANTIATE_CLASS(RegularizerAsLossLayer);

}  // namespace caffe
