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
Dtype DistanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data_0 = bottom[0]->gpu_data();
  const Dtype* bottom_data_1 = bottom[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* diff_data = difference_.gpu_data();
  const Dtype* diff_sq_data = difference_squared_.gpu_data();

  int count = bottom[0]->count();

  switch (this->layer_param_.distance_param().distance()) {
  case DistanceParameter_Distance_Squared:
    caffe_gpu_copy(count, bottom_data_0, difference_.mutable_gpu_data());
    caffe_gpu_axpy(count, Dtype(-1), bottom_data_1,
        difference_.mutable_gpu_data());

    caffe_gpu_mul(count, diff_data, diff_data,
        difference_squared_.mutable_gpu_data());

    // project diff squared on w
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
        diff_sq_data, weight, (Dtype)0., top_data);
    break;
  case DistanceParameter_Distance_Abs:
  default:
    LOG(FATAL) << "Unknown Distance";
  }

  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);

    //LOG(ERROR) << "b: " << this->blobs_[1]->gpu_data()[0];
  }
  return Dtype(0);
}

template <typename Dtype>
void DistanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* diff_sq_data = difference_squared_.gpu_data();
  const Dtype* diff_data = difference_.gpu_data();

  int count = (*bottom)[0]->count();
  int num = (*bottom)[0]->num();
 
  switch (this->layer_param_.distance_param().distance()) {
  case DistanceParameter_Distance_Squared:
    // Gradient with respect to weight, squared diff
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, diff_sq_data, (Dtype)0., this->blobs_[0]->mutable_gpu_diff());
    break;
  case DistanceParameter_Distance_Abs:
  default:
    LOG(FATAL) << "Unknown Distance";
  }
 
  if (bias_term_) {
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()), (Dtype)0.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    switch (this->layer_param_.distance_param().distance()) {
    case DistanceParameter_Distance_Squared:

      // Gradient with respect to bottom data, squared diff
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)2.,
          top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
          (*bottom)[0]->mutable_gpu_diff());
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)-2.,
          top_diff, this->blobs_[0]->gpu_data(), (Dtype)0.,
          (*bottom)[1]->mutable_gpu_diff());


      caffe_gpu_mul(count, diff_data, (*bottom)[0]->gpu_diff(),
          (*bottom)[0]->mutable_gpu_diff());
      caffe_gpu_mul(count, diff_data, (*bottom)[1]->gpu_diff(),
          (*bottom)[1]->mutable_gpu_diff());
      break;

    case DistanceParameter_Distance_Abs:
    default:
      LOG(FATAL) << "Unknown Distance";
    }

  }
}

INSTANTIATE_CLASS(DistanceLayer);

}  // namespace caffe
