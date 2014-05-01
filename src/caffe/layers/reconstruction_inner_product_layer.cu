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
Dtype ReconstructionInnerProductLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  // compute the reconstruction error and return that as loss value.
  caffe_gpu_copy<Dtype>(bottom[0]->count(),
      bottom_data, difference_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
      top_data, weight, (Dtype)-1., difference_.mutable_gpu_data());
  Dtype loss;
  caffe_gpu_dot<Dtype>(difference_.count(),
      difference_.gpu_data(), difference_.gpu_data(), &loss);
  return loss;
}

template <typename Dtype>
void ReconstructionInnerProductLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  // Gradient with respect to weight
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
      top_diff, bottom_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());

  // now add the reconstruction cost's gradient

  const Dtype* weight = this->blobs_[0]->gpu_data();
  // compute W^T W
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, N_, (Dtype)1.,
      weight, weight, (Dtype)0., w_Tw_.mutable_gpu_data());

  // compute X^T X
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1.,
      bottom_data, bottom_data, (Dtype)0., x_Tx_.mutable_gpu_data());

  // compute X^T X W^T W
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, K_, K_, (Dtype)1.,
      x_Tx_.gpu_data(), w_Tw_.gpu_data(), (Dtype)0.,
      x_Txw_Tw_.mutable_gpu_data());

  // now add 2 W X^T X W^T W to the weight diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)2.,
      weight, x_Txw_Tw_.gpu_data(),
      (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

  // now add 2 W W^T W X^T X to the weight diff.
  // Note that W^T W X^T X is the transpose of X^T X W^T W
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, K_, K_, (Dtype)2.,
      weight, x_Txw_Tw_.gpu_data(),
      (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

  // now add -4 W X^T X to weight diff
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, K_, K_, (Dtype)-4.,
      weight, x_Tx_.gpu_data(), (Dtype)1., this->blobs_[0]->mutable_gpu_diff());

  if (propagate_down) {
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
        (*bottom)[0]->mutable_gpu_diff());

    // now add the gradient of the reconstruction term w.r.t. bottom
    // add 2 X to the mutable_gpu_diff
    caffe_gpu_axpby<Dtype>((*bottom)[0]->count(),
        (Dtype)2., (*bottom)[0]->gpu_data(),
        (Dtype)1., (*bottom)[0]->mutable_gpu_diff());

    // compute -4 W^T W + 2 W^T W W^T W
    // -- note w_Tw_ is no longer W^T W :(
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, K_, K_,
        (Dtype)2., w_Tw_.gpu_data(), w_Tw_.gpu_data(), (Dtype)-4.,
        w_Tw_.mutable_gpu_data());

    // multiply the above thing by X and add it to mutable_gpu_diff
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_,
        (Dtype)1., (*bottom)[0]->gpu_data(), w_Tw_.gpu_data(),
        (Dtype)1., (*bottom)[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_CLASS(ReconstructionInnerProductLayer);

}  // namespace caffe
