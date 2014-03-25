// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
using std::vector;

template<typename Dtype>
Dtype RegularizerAsLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Blob<Dtype>* bottom_data = bottom[0];
  if (bottom_data->count() > 0) {
    CUDA_CHECK(
        cudaMemset(bottom_data->mutable_gpu_diff(), 0,
                   bottom_data->count() * sizeof(Dtype)));
    Dtype loss = 0;
    for (int i = 0; i < num_regularizers_; ++i) {
      loss += regularizers_[i]->Regularize_gpu(bottom_data);
    }
    int num = bottom_data->num();
    // Scale down gradient
    caffe_gpu_scal<Dtype>(bottom_data->count(), Dtype(1) / num,
                          bottom_data->mutable_gpu_diff());
    return loss / num;
  }
  return Dtype(0);
}

template<typename Dtype>
void RegularizerAsLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  return;
}

INSTANTIATE_CLASS(RegularizerAsLossLayer);

}  // namespace caffe
