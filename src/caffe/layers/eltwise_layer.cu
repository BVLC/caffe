// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype EltwiseLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    caffe_gpu_mul(count, bottom[0]->gpu_data(),
        bottom[1]->gpu_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    caffe_gpu_set(count, Dtype(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom->size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1.)) {
          caffe_gpu_copy(count, top_diff, bottom_diff);
        } else {
          caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

INSTANTIATE_CLASS(EltwiseLayer);


}  // namespace caffe
