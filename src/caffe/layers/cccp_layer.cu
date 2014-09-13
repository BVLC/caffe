// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CCCPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_offset = num_output_ /group_ * channels_ / group_;
  const int bottom_group_offset = width_ * height_ * channels_ / group_;
  const int top_group_offset = width_ * height_ * num_output_ / group_;

  for (int n = 0; n < num_; ++n) {
    for (int g = 0; g < group_; ++g) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_ / group_,
          width_ * height_, channels_ / group_, (Dtype)1.,
          weight + g * weight_offset,
          bottom_data + bottom[0]->offset(n) + g * bottom_group_offset,
          (Dtype)0., top_data + (*top)[0]->offset(n) + g * top_group_offset);
    }
    if (bias_term_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
          width_ * height_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          reinterpret_cast<const Dtype*>(bias_multiplier_.gpu_data()),
          (Dtype)1., top_data + (*top)[0]->offset(n));
    }
  }
}

template <typename Dtype>
void CCCPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  Dtype* bias_diff = NULL;
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();

  const int weight_offset = num_output_ / group_ * channels_ / group_;
  const int bottom_group_offset = width_ * height_ * channels_ / group_;
  const int top_group_offset = width_ * height_ * num_output_ / group_;

  // Gradient with respect to bias
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    CUDA_CHECK(cudaMemset(bias_diff, 0,
            sizeof(Dtype) * this->blobs_[1]->count()));
    for (int n = 0; n < num_; ++n) {
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_,
          width_ * height_, (Dtype)1., top_diff + top[0]->offset(n),
          reinterpret_cast<const Dtype*>(bias_multiplier_.gpu_data()),
          (Dtype)1., bias_diff);
    }
  }

  if (this->param_propagate_down_[0] || propagate_down[0]) {
    if (this->param_propagate_down_[0]) {
      CUDA_CHECK(cudaMemset(weight_diff, 0,
              sizeof(Dtype) * this->blobs_[0]->count()));
    }
    for (int n = 0; n < num_; ++n) {
      if (this->param_propagate_down_[0]) {
        // The gradient will be accumulated
        for (int g = 0; g < group_; ++g) {
          // Gradient with respect to weight
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num_output_ / group_,
              channels_ / group_, width_ * height_, (Dtype)1.,
              top_diff + top[0]->offset(n) + g * top_group_offset,
              bottom_data + (*bottom)[0]->offset(n) + g * bottom_group_offset,
              (Dtype)1., weight_diff + g * weight_offset);
        }
      }
      if (propagate_down[0]) {
        for (int g = 0; g < group_; ++g) {
          // Gradient w.r.t. bottom data if necessary
          caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_ / group_,
              width_ * height_, num_output_ / group_, (Dtype)1.,
              weight + g * weight_offset,
              top_diff + top[0]->offset(n) + g * top_group_offset, (Dtype)0.,
              bottom_diff + (*bottom)[0]->offset(n) + g * bottom_group_offset);
        }
      }
    }
  }
}

INSTANTIATE_CLASS(CCCPLayer);

}  // namespace caffe
