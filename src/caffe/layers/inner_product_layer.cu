// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
  return Dtype(0);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
  if (bias_term_) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)0., this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(InnerProductLayer);

}  // namespace caffe
