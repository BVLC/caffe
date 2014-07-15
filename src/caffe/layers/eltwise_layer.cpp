// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EltwiseLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Layer<Dtype>::SetUp(bottom, top);
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK_EQ(num, bottom[i]->num());
    CHECK_EQ(channels, bottom[i]->channels());
    CHECK_EQ(height, bottom[i]->height());
    CHECK_EQ(width, bottom[i]->width());
  }
  (*top)[0]->Reshape(num, channels, height, width);
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
    }
  }
}

template <typename Dtype>
Dtype EltwiseLayer<Dtype>::Forward(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int count = (*top)[0]->count();
  Dtype* top_data = (*top)[0]->mutable_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    this->device_->mul(count, bottom[0]->const_data(),
                       bottom[1]->const_data(), top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      this->device_->mul(count, top_data, bottom[i]->const_data(), top_data);
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    this->device_->set(count, Dtype(0), top_data);
    // TODO(shelhamer) does BLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      this->device_->axpy(count, coeffs_[i], bottom[i]->const_data(),
                          top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
  return Dtype(0.);
}

template <typename Dtype>
void EltwiseLayer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const int count = top[0]->count();
  const Dtype* top_data = top[0]->const_data();
  const Dtype* top_diff = top[0]->const_diff();
  for (int i = 0; i < bottom->size(); ++i) {
    if (propagate_down[i]) {
      const Dtype* bottom_data = (*bottom)[i]->const_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_diff();
      switch (op_) {
      case EltwiseParameter_EltwiseOp_PROD:
        this->device_->div(count, top_data, bottom_data, bottom_diff);
        this->device_->mul(count, bottom_diff, top_diff, bottom_diff);
        break;
      case EltwiseParameter_EltwiseOp_SUM:
        if (coeffs_[i] == Dtype(1)) {
          this->device_->copy(count, top_diff, bottom_diff);
        } else {
          this->device_->scale(count, coeffs_[i], top_diff, bottom_diff);
        }
        break;
      default:
        LOG(FATAL) << "Unknown elementwise operation.";
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(EltwiseLayer);


}  // namespace caffe
